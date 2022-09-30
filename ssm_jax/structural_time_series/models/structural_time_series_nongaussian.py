from collections import OrderedDict
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit, lax
from ssm_jax.distributions import InverseWishart as IW
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import blackjax
import jax.random as jr
from jax.tree_util import tree_map
from ssm_jax.abstractions import SSM
from ssm_jax.linear_gaussian_ssm.inference import (
    LGSSMParams, lgssm_filter, lgssm_smoother, lgssm_posterior_sample)
from ssm_jax.structural_time_series.new_parameters import (
    to_unconstrained, from_unconstrained, log_det_jac_constrain, ParameterProperties)
from ssm_jax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax.bijectors as tfb
from tqdm.auto import trange


def _set_prior(input_prior, default_prior):
    return input_prior if input_prior is not None else default_prior


class StructuralTimeSeries():

    def __init__(self,
                 components,
                 observed_timeseries,
                 observation_distribution_family='Gaussian',
                 observation_covariance=None,
                 observation_covariance_prior=None,
                 name='StructuralTimeSeries'):

        assert observation_distribution_family in ['Gaussian', 'Poisson']

        _dim = observed_timeseries.shape
        self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
        obs_scale = jnp.std(jnp.abs(jnp.diff(observed_timeseries, axis=0)), axis=0).mean()
        self.observation_distribution_family = observation_distribution_family
        self.name = name

        # Save parameters of the STS model:
        self.initial_state_priors = OrderedDict()
        self.transition_matrices = OrderedDict()
        self.transition_covariances = OrderedDict()
        self.transition_covariance_priors = OrderedDict()
        self.cov_spars_matrices = OrderedDict()
        self.observation_matrices = OrderedDict()

        if observation_distribution_family == 'Gaussian':
            self.observation_covariance_prior = _set_prior(
                observation_covariance_prior,
                IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs))
                )
            if observation_covariance is not None:
                self.observation_covariance = observation_covariance
            else:
                self.observation_covariance = 1e-4*obs_scale**2*jnp.eye(self.dim_obs)

        self.observation_regression_weights = None
        self.observation_regression_weights_prior = None

        # Aggregate components
        for c in components:
            if isinstance(c, LinearRegression):
                self.observation_regression_weights = c.weights_prior.mode()
                self.observation_regression_weights_prior = c.weights_prior
            elif isinstance(c, STSLatentComponent):
                self.transition_matrices.update(c.transition_matrix)
                self.observation_matrices.update(c.observation_matrix)
                self.initial_state_priors.update(c.initial_state_prior)
                self.transition_covariances.update(c.transition_covariance)
                self.transition_covariance_priors.update(c.transition_covariance_prior)
                self.cov_spars_matrices.update(c.cov_spars_matrix)

    def as_ssm(self):
        """Formulate the STS model as a linear Gaussian state space model:

        p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
        p(y_t | z_t) = N(y_t | H_t z_t, R_t)
        p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})

        F_t, H_t are fixed known matrices,
        the convariance matrices, Q and R, are random variables to be learned,
        the regression coefficient matrix B is also unknown random matrix
        if the STS model includes an regression component
        """
        sts_ssm = StructuralTimeSeriesSSM(self.transition_matrices,
                                          self.observation_matrices,
                                          self.initial_state_priors,
                                          self.transition_covariances,
                                          self.transition_covariance_priors,
                                          self.observation_covariance,
                                          self.observation_covariance_prior,
                                          self.cov_spars_matrices,
                                          self.observation_regression_weights,
                                          self.observation_regression_weights_prior,
                                          self.observation_distribution_family)
        return sts_ssm

    def posterior_sample(self, key, observed_time_series, sts_params, inputs=None):
        @jit
        def _single_sample(sts_param):
            sts_ssm = StructuralTimeSeriesSSM(self.transition_matrices,
                                              self.observation_matrices,
                                              self.initial_state_priors,
                                              sts_param['dynamics_covariances'],
                                              self.transition_covariance_priors,
                                              sts_param['emission_covariance'],
                                              self.observation_covariance_prior,
                                              self.cov_spars_matrices,
                                              sts_param['regression_weights'],
                                              self.observation_regression_weights_prior,
                                              self.observation_distribution_family)
            ts_means, ts = sts_ssm.posterior_sample(key, observed_time_series, inputs)
            return [ts_means, ts]
        samples = vmap(_single_sample)(sts_params)
        return {'means': samples[0], 'observations': samples[1]}

    def fit_hmc(self, key, sample_size, observed_time_series, inputs=None,
                warmup_steps=500, num_integration_steps=30):
        """Sampling parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            covariance matrix of observation,
            regression coefficient matrix (if the model has inputs and a regression component)
        """
        sts_ssm = self.as_ssm()
        param_samps = sts_ssm.fit_hmc(key, sample_size, observed_time_series, inputs,
                                      warmup_steps, num_integration_steps)
        return param_samps

    def forecast(self, key, observed_time_series, sts_params, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        # Set the new initial_state_prior to be at the last observation
        @jit
        def _single_sample(sts_param):
            sts_ssm = StructuralTimeSeriesSSM(self.transition_matrices,
                                              self.observation_matrices,
                                              self.initial_state_priors,
                                              sts_param['dynamics_covariances'],
                                              self.transition_covariance_priors,
                                              sts_param['emission_covariance'],
                                              self.observation_covariance_prior,
                                              self.cov_spars_matrices,
                                              sts_param['regression_weights'],
                                              self.observation_regression_weights_prior,
                                              self.observation_distribution_family)
            means, covs, ts = sts_ssm.forecast(key, observed_time_series, num_forecast_steps,
                                               past_inputs, forecast_inputs)
            return [means, covs, ts]

        samples = vmap(_single_sample)(sts_params)
        return {'means': samples[0], 'covariances': samples[1], 'observations': samples[2]}


##########################
# Learning and inference #
##########################


class StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearGaussianSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    The covariance matrix of the dynamics model takes the form:
    R @ Q, where Q is a dense matrix (blockwise diagonal),
    and R is the sparsing matrix. For example,
    for an STS model for a 1-d time series with a local linear component
    and a seasonal component with 4 seasons:
                                        | 1, 0, 0 |
                | v1,   0,  0 |         | 0, 1, 0 |
            Q = |  0,  v2,  0 |,    R = | 0, 0, 1 |
                |  0,   0, v3 |         | 0, 0, 0 |
                                        | 0, 0, 0 |
    """

    def __init__(self,
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariances,
                 component_transition_covariance_priors,
                 observation_covariance,
                 observation_covariance_prior,
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None,
                 observation_distribution_family='Gaussian'):

        # Set parameters for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in component_initial_state_priors.values()])
        self.initial_covariance = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in component_initial_state_priors.values()])

        # Set parameters of the dynamics model of the LinearGaussainSSM model
        self.dynamics_matrix = jsp.linalg.block_diag(*component_transition_matrices.values())
        self.state_dim = self.dynamics_matrix.shape[-1]
        self.dynamics_bias = jnp.zeros(self.state_dim)
        dynamics_covariance_props = OrderedDict()
        for c in component_transition_covariance_priors.keys():
            dynamics_covariance_props[c] = ParameterProperties(
                trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
        self.spars_matrix = cov_spars_matrices

        # Set parameters of the emission model of the LinearGaussianSSM model
        self.emission_matrix = jnp.concatenate(list(component_observation_matrices.values()), axis=1)
        self.emission_dim = self.emission_matrix.shape[0]
        if observation_regression_weights is not None:
            emission_input_weights = observation_regression_weights
            shape_in = emission_input_weights.shape
            size_in = emission_input_weights.size
            emission_input_weights_props = ParameterProperties(
                trainable=True,
                constrainer=tfb.Reshape(event_shape_out=shape_in, event_shape_in=(size_in,))
                )
            emission_input_weights_prior = observation_regression_weights_prior
        else:
            emission_input_weights = jnp.zeros((self.emission_dim, 0))
            emission_input_weights_props = ParameterProperties(trainable=False)
            emission_input_weights_prior = None
        self.emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))

        # Parameters, their properties, and priors of the SSM model
        self.params = {'dynamics_covariances': component_transition_covariances,
                       'emission_covariance': observation_covariance,
                       'regression_weights': emission_input_weights}

        self.param_props = {'dynamics_covariances': dynamics_covariance_props,
                            'emission_covariance': emission_covariance_props,
                            'regression_weights': emission_input_weights_props}

        self.priors = {'dynamics_covariances': component_transition_covariance_priors,
                       'emission_covariance': observation_covariance_prior,
                       'regression_weights': emission_input_weights_prior}

    def log_prior(self, params):
        lp = jnp.array([cov_prior.log_prob(cov) for cov, cov_prior in
                        zip(params['dynamics_covariances'].values(),
                            self.priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        if params['regression_weights'].size > 0:
            lp += self.priors['regression_weights'].log_prob(params['regression_weights'])
        return lp

    # Set component distributions of SSM
    def initial_distribution(self):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
        multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        return MVN(self.emission_matrix @ state + self.params['regression_weights'] @ inputs,
                   self.params['emission_covariance'])

    def sample(self, key, num_timesteps, inputs=None):
        """Sample a sequence of latent states and emissions.
        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        dim_comp = comp_cov.shape[-1]
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())

        def _step(prev_state, args):
            key, input = args
            key1, key2 = jr.split(key, 2)
            state = prev_state + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)
            emission = self.emission_distribution(state, input).sample(seed=key2)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)
        initial_emission = self.emission_distribution(initial_state, inputs[0]).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, inputs[1:]))

        # Concatenate the initial state and emission with the following ones
        states = jnp.concatenate((jnp.expand_dims(initial_state, 0), next_states))
        emissions = jnp.concatenate((jnp.expand_dims(initial_emission, 0), next_emissions))
        return states, emissions

    def _to_lgssm_params(self, params):
        comp_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        obs_cov = params['emission_covariance']
        emission_input_weights = params['regression_weights']
        input_dim = emission_input_weights.shape[-1]
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=self.dynamics_matrix,
                           dynamics_input_weights=jnp.zeros((self.state_dim, input_dim)),
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=spars_cov,
                           emission_matrix=self.emission_matrix,
                           emission_input_weights=emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=obs_cov)

    def marginal_log_prob(self, emissions, inputs=None, params=None):
        """Compute log marginal likelihood of observations."""
        if params is None:
            # Compute marginal log prob using current parameter
            lgssm_params = self._to_lgssm_params(self.params)
        else:
            # Compute marginal log prob using given parameter
            lgssm_params = self._to_lgssm_params(params)

        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.filtered_means, filtered_posterior.filtered_covariances

    def smoother(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        smoothed_posterior = lgssm_smoother(lgssm_params, emissions, inputs)
        return smoothed_posterior.smoothed_means, smoothed_posterior.smoothed_covariances

    def posterior_sample(self, key, observed_time_series, inputs=None):
        num_timesteps, dim_obs = observed_time_series.shape
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        obs_cov = self.params['emission_covariance']
        lgssm_params = self._to_lgssm_params(self.params)
        ll, states = lgssm_posterior_sample(key, lgssm_params, observed_time_series, inputs)
        obs_means = states @ self.emission_matrix.T + inputs @ self.params['regression_weights'].T
        obs = obs_means + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key, sample_shape=num_timesteps)
        return obs_means, obs

    def fit_hmc(self,
                key,
                sample_size,
                emissions,
                inputs=None,
                warmup_steps=500,
                num_integration_steps=30):

        def logprob(trainable_unc_params):
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            logprob,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)

        @jit
        def _step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            unc_sample = next_state.position
            return next_state, unc_sample

        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state, unc_sample = _step(current_state, next(keys))
            sample = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        param_samples = tree_map(lambda x, *y: jnp.array([x] + [i for i in y]),
                                 param_samples[0], *param_samples[1:])
        return param_samples

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        """Forecast the time series"""

        if forecast_inputs is None:
            forecast_inputs = jnp.zeros((num_forecast_steps, 0))
        weights = self.params['regression_weights']
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        obs_cov = self.params['emission_covariance']
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        dim_obs = observed_time_series.shape[-1]
        dim_comp = comp_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + spars_cov
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)

        def _step(prev_params, args):
            key, forecast_input = args
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            marginal_mean = self.emission_matrix @ prev_mean + weights @ forecast_input
            marginal_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T + obs_cov
            obs = self.emission_matrix @ prev_state + weights @ forecast_input\
                + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key2)

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)

            return (next_mean, next_cov, next_state), (marginal_mean, marginal_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        _, (ts_means, ts_covs, ts) = lax.scan(_step, initial_params, (keys, forecast_inputs))

        return ts_means, ts_covs, ts


class STSGaussian(StructuralTimeSeriesSSM):
    """Formulate the structual time series(STS) model into a LinearGaussianSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    The covariance matrix of the dynamics model takes the form:
    R @ Q, where Q is a dense matrix (blockwise diagonal),
    and R is the sparsing matrix. For example,
    for an STS model for a 1-d time series with a local linear component
    and a seasonal component with 4 seasons:
                                        | 1, 0, 0 |
                | v1,   0,  0 |         | 0, 1, 0 |
            Q = |  0,  v2,  0 |,    R = | 0, 0, 1 |
                |  0,   0, v3 |         | 0, 0, 0 |
                                        | 0, 0, 0 |
    """

    def __init__(self,
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariances,
                 component_transition_covariance_priors,
                 observation_covariance,
                 observation_covariance_prior,
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None,
                 observation_distribution_family='Gaussian'):

        # Set parameters for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in component_initial_state_priors.values()])
        self.initial_covariance = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in component_initial_state_priors.values()])

        # Set parameters of the dynamics model of the LinearGaussainSSM model
        self.dynamics_matrix = jsp.linalg.block_diag(*component_transition_matrices.values())
        self.state_dim = self.dynamics_matrix.shape[-1]
        self.dynamics_bias = jnp.zeros(self.state_dim)
        dynamics_covariance_props = OrderedDict()
        for c in component_transition_covariance_priors.keys():
            dynamics_covariance_props[c] = ParameterProperties(
                trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
        self.spars_matrix = cov_spars_matrices

        # Set parameters of the emission model of the LinearGaussianSSM model
        self.emission_matrix = jnp.concatenate(list(component_observation_matrices.values()), axis=1)
        self.emission_dim = self.emission_matrix.shape[0]
        if observation_regression_weights is not None:
            emission_input_weights = observation_regression_weights
            shape_in = emission_input_weights.shape
            size_in = emission_input_weights.size
            emission_input_weights_props = ParameterProperties(
                trainable=True,
                constrainer=tfb.Reshape(event_shape_out=shape_in, event_shape_in=(size_in,))
                )
            emission_input_weights_prior = observation_regression_weights_prior
        else:
            emission_input_weights = jnp.zeros((self.emission_dim, 0))
            emission_input_weights_props = ParameterProperties(trainable=False)
            emission_input_weights_prior = None
        self.emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))

        # Parameters, their properties, and priors of the SSM model
        self.params = {'dynamics_covariances': component_transition_covariances,
                       'emission_covariance': observation_covariance,
                       'regression_weights': emission_input_weights}

        self.param_props = {'dynamics_covariances': dynamics_covariance_props,
                            'emission_covariance': emission_covariance_props,
                            'regression_weights': emission_input_weights_props}

        self.priors = {'dynamics_covariances': component_transition_covariance_priors,
                       'emission_covariance': observation_covariance_prior,
                       'regression_weights': emission_input_weights_prior}

    def log_prior(self, params):
        lp = jnp.array([cov_prior.log_prob(cov) for cov, cov_prior in
                        zip(params['dynamics_covariances'].values(),
                            self.priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        if params['regression_weights'].size > 0:
            lp += self.priors['regression_weights'].log_prob(params['regression_weights'])
        return lp

    # Set component distributions of SSM
    def initial_distribution(self):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
        multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        return MVN(self.emission_matrix @ state + self.params['regression_weights'] @ inputs,
                   self.params['emission_covariance'])

    def sample(self, key, num_timesteps, inputs=None):
        """Sample a sequence of latent states and emissions.
        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        dim_comp = comp_cov.shape[-1]
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())

        def _step(prev_state, args):
            key, input = args
            key1, key2 = jr.split(key, 2)
            state = prev_state + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)
            emission = self.emission_distribution(state, input).sample(seed=key2)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)
        initial_emission = self.emission_distribution(initial_state, inputs[0]).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, inputs[1:]))

        # Concatenate the initial state and emission with the following ones
        states = jnp.concatenate((jnp.expand_dims(initial_state, 0), next_states))
        emissions = jnp.concatenate((jnp.expand_dims(initial_emission, 0), next_emissions))
        return states, emissions

    def _to_lgssm_params(self, params):
        comp_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        obs_cov = params['emission_covariance']
        emission_input_weights = params['regression_weights']
        input_dim = emission_input_weights.shape[-1]
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=self.dynamics_matrix,
                           dynamics_input_weights=jnp.zeros((self.state_dim, input_dim)),
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=spars_cov,
                           emission_matrix=self.emission_matrix,
                           emission_input_weights=emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=obs_cov)

    def marginal_log_prob(self, emissions, inputs=None, params=None):
        """Compute log marginal likelihood of observations."""
        if params is None:
            # Compute marginal log prob using current parameter
            lgssm_params = self._to_lgssm_params(self.params)
        else:
            # Compute marginal log prob using given parameter
            lgssm_params = self._to_lgssm_params(params)

        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.filtered_means, filtered_posterior.filtered_covariances

    def smoother(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        smoothed_posterior = lgssm_smoother(lgssm_params, emissions, inputs)
        return smoothed_posterior.smoothed_means, smoothed_posterior.smoothed_covariances

    def posterior_sample(self, key, observed_time_series, inputs=None):
        num_timesteps, dim_obs = observed_time_series.shape
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        obs_cov = self.params['emission_covariance']
        lgssm_params = self._to_lgssm_params(self.params)
        ll, states = lgssm_posterior_sample(key, lgssm_params, observed_time_series, inputs)
        obs_means = states @ self.emission_matrix.T + inputs @ self.params['regression_weights'].T
        obs = obs_means + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key, sample_shape=num_timesteps)
        return obs_means, obs

    def fit_hmc(self,
                key,
                sample_size,
                emissions,
                inputs=None,
                warmup_steps=500,
                num_integration_steps=30):

        def logprob(trainable_unc_params):
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            logprob,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)

        @jit
        def _step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            unc_sample = next_state.position
            return next_state, unc_sample

        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state, unc_sample = _step(current_state, next(keys))
            sample = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        param_samples = tree_map(lambda x, *y: jnp.array([x] + [i for i in y]),
                                 param_samples[0], *param_samples[1:])
        return param_samples

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        """Forecast the time series"""

        if forecast_inputs is None:
            forecast_inputs = jnp.zeros((num_forecast_steps, 0))
        weights = self.params['regression_weights']
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        obs_cov = self.params['emission_covariance']
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        dim_obs = observed_time_series.shape[-1]
        dim_comp = comp_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + spars_cov
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)

        def _step(prev_params, args):
            key, forecast_input = args
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            marginal_mean = self.emission_matrix @ prev_mean + weights @ forecast_input
            marginal_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T + obs_cov
            obs = self.emission_matrix @ prev_state + weights @ forecast_input\
                + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key2)

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)

            return (next_mean, next_cov, next_state), (marginal_mean, marginal_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        _, (ts_means, ts_covs, ts) = lax.scan(_step, initial_params, (keys, forecast_inputs))

        return ts_means, ts_covs, ts


class SSMPoisson(StructuralTimeSeriesSSM):
    """Formulate the structual time series(STS) model into a LinearGaussianSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    The covariance matrix of the dynamics model takes the form:
    R @ Q, where Q is a dense matrix (blockwise diagonal),
    and R is the sparsing matrix. For example,
    for an STS model for a 1-d time series with a local linear component
    and a seasonal component with 4 seasons:
                                        | 1, 0, 0 |
                | v1,   0,  0 |         | 0, 1, 0 |
            Q = |  0,  v2,  0 |,    R = | 0, 0, 1 |
                |  0,   0, v3 |         | 0, 0, 0 |
                                        | 0, 0, 0 |
    """

    def __init__(self,
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariances,
                 component_transition_covariance_priors,
                 observation_covariance,
                 observation_covariance_prior,
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None,
                 observation_distribution_family='Gaussian'):

        # Set parameters for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in component_initial_state_priors.values()])
        self.initial_covariance = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in component_initial_state_priors.values()])

        # Set parameters of the dynamics model of the LinearGaussainSSM model
        self.dynamics_matrix = jsp.linalg.block_diag(*component_transition_matrices.values())
        self.state_dim = self.dynamics_matrix.shape[-1]
        self.dynamics_bias = jnp.zeros(self.state_dim)
        dynamics_covariance_props = OrderedDict()
        for c in component_transition_covariance_priors.keys():
            dynamics_covariance_props[c] = ParameterProperties(
                trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
        self.spars_matrix = cov_spars_matrices

        # Set parameters of the emission model of the LinearGaussianSSM model
        self.emission_matrix = jnp.concatenate(list(component_observation_matrices.values()), axis=1)
        self.emission_dim = self.emission_matrix.shape[0]
        if observation_regression_weights is not None:
            emission_input_weights = observation_regression_weights
            shape_in = emission_input_weights.shape
            size_in = emission_input_weights.size
            emission_input_weights_props = ParameterProperties(
                trainable=True,
                constrainer=tfb.Reshape(event_shape_out=shape_in, event_shape_in=(size_in,))
                )
            emission_input_weights_prior = observation_regression_weights_prior
        else:
            emission_input_weights = jnp.zeros((self.emission_dim, 0))
            emission_input_weights_props = ParameterProperties(trainable=False)
            emission_input_weights_prior = None
        self.emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))

        # Parameters, their properties, and priors of the SSM model
        self.params = {'dynamics_covariances': component_transition_covariances,
                       'emission_covariance': observation_covariance,
                       'regression_weights': emission_input_weights}

        self.param_props = {'dynamics_covariances': dynamics_covariance_props,
                            'emission_covariance': emission_covariance_props,
                            'regression_weights': emission_input_weights_props}

        self.priors = {'dynamics_covariances': component_transition_covariance_priors,
                       'emission_covariance': observation_covariance_prior,
                       'regression_weights': emission_input_weights_prior}

    def log_prior(self, params):
        lp = jnp.array([cov_prior.log_prob(cov) for cov, cov_prior in
                        zip(params['dynamics_covariances'].values(),
                            self.priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        if params['regression_weights'].size > 0:
            lp += self.priors['regression_weights'].log_prob(params['regression_weights'])
        return lp

    # Set component distributions of SSM
    def initial_distribution(self):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
        multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        return MVN(self.emission_matrix @ state + self.params['regression_weights'] @ inputs,
                   self.params['emission_covariance'])

    def sample(self, key, num_timesteps, inputs=None):
        """Sample a sequence of latent states and emissions.
        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        dim_comp = comp_cov.shape[-1]
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())

        def _step(prev_state, args):
            key, input = args
            key1, key2 = jr.split(key, 2)
            state = prev_state + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)
            emission = self.emission_distribution(state, input).sample(seed=key2)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution().sample(seed=key1)
        initial_emission = self.emission_distribution(initial_state, inputs[0]).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, inputs[1:]))

        # Concatenate the initial state and emission with the following ones
        states = jnp.concatenate((jnp.expand_dims(initial_state, 0), next_states))
        emissions = jnp.concatenate((jnp.expand_dims(initial_emission, 0), next_emissions))
        return states, emissions

    def _to_lgssm_params(self, params):
        comp_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        obs_cov = params['emission_covariance']
        emission_input_weights = params['regression_weights']
        input_dim = emission_input_weights.shape[-1]
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=self.dynamics_matrix,
                           dynamics_input_weights=jnp.zeros((self.state_dim, input_dim)),
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=spars_cov,
                           emission_matrix=self.emission_matrix,
                           emission_input_weights=emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=obs_cov)

    def marginal_log_prob(self, emissions, inputs=None, params=None):
        """Compute log marginal likelihood of observations."""
        if params is None:
            # Compute marginal log prob using current parameter
            lgssm_params = self._to_lgssm_params(self.params)
        else:
            # Compute marginal log prob using given parameter
            lgssm_params = self._to_lgssm_params(params)

        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.filtered_means, filtered_posterior.filtered_covariances

    def smoother(self, emissions, inputs=None):
        lgssm_params = self._to_lgssm_params(self.params)
        smoothed_posterior = lgssm_smoother(lgssm_params, emissions, inputs)
        return smoothed_posterior.smoothed_means, smoothed_posterior.smoothed_covariances

    def posterior_sample(self, key, observed_time_series, inputs=None):
        num_timesteps, dim_obs = observed_time_series.shape
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        obs_cov = self.params['emission_covariance']
        lgssm_params = self._to_lgssm_params(self.params)
        ll, states = lgssm_posterior_sample(key, lgssm_params, observed_time_series, inputs)
        obs_means = states @ self.emission_matrix.T + inputs @ self.params['regression_weights'].T
        obs = obs_means + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key, sample_shape=num_timesteps)
        return obs_means, obs

    def fit_hmc(self,
                key,
                sample_size,
                emissions,
                inputs=None,
                warmup_steps=500,
                num_integration_steps=30):

        def logprob(trainable_unc_params):
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            logprob,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)

        @jit
        def _step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            unc_sample = next_state.position
            return next_state, unc_sample

        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state, unc_sample = _step(current_state, next(keys))
            sample = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        param_samples = tree_map(lambda x, *y: jnp.array([x] + [i for i in y]),
                                 param_samples[0], *param_samples[1:])
        return param_samples

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        """Forecast the time series"""

        if forecast_inputs is None:
            forecast_inputs = jnp.zeros((num_forecast_steps, 0))
        weights = self.params['regression_weights']
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        obs_cov = self.params['emission_covariance']
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        dim_obs = observed_time_series.shape[-1]
        dim_comp = comp_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, past_inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + spars_cov
        initial_state = MVN(initial_mean, initial_cov).sample(seed=key)

        def _step(prev_params, args):
            key, forecast_input = args
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            marginal_mean = self.emission_matrix @ prev_mean + weights @ forecast_input
            marginal_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T + obs_cov
            obs = self.emission_matrix @ prev_state + weights @ forecast_input\
                + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key2)

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)

            return (next_mean, next_cov, next_state), (marginal_mean, marginal_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        _, (ts_means, ts_covs, ts) = lax.scan(_step, initial_params, (keys, forecast_inputs))

        return ts_means, ts_covs, ts
