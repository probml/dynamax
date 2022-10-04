import blackjax
from collections import OrderedDict
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from ssm_jax.abstractions import SSM
from ssm_jax.cond_moments_gaussian_filter.containers import EKFParams
from ssm_jax.cond_moments_gaussian_filter.inference import (
    iterated_conditional_moments_gaussian_filter as cmgf,)
from ssm_jax.linear_gaussian_ssm.inference import (
    LGSSMParams, lgssm_filter, lgssm_smoother, lgssm_posterior_sample
    )
from ssm_jax.structural_time_series.new_parameters import (
    to_unconstrained, from_unconstrained, log_det_jac_constrain, ParameterProperties
    )
from ssm_jax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN, Poisson
    )
from tqdm.auto import trange


class _StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    The covariance matrix of the latent dynamics model takes the form:
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
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None):

        # Set parameters for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in component_initial_state_priors.values()]
            )
        self.initial_covariance = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in component_initial_state_priors.values()]
            )
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

        # Parameters, their properties, and priors of the SSM model
        self.params = {'dynamics_covariances': component_transition_covariances,
                       'regression_weights': emission_input_weights}

        self.param_props = {'dynamics_covariances': dynamics_covariance_props,
                            'regression_weights': emission_input_weights_props}

        self.priors = {'dynamics_covariances': component_transition_covariance_priors,
                       'regression_weights': emission_input_weights_prior}

    def log_prior(self, params):
        lp = jnp.array([
            cov_prior.log_prob(cov) for cov, cov_prior in zip(
                params['dynamics_covariances'].values(), self.priors['dynamics_covariances'].values()
                )]).sum()
        if params['regression_weights'].size > 0:
            lp += self.priors['regression_weights'].log_prob(params['regression_weights'])
        return lp

    # Instantiate distributions of the SSM model
    def initial_distribution(self):
        """Gaussian distribution of the initial state of the SSM model.
        """
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state):
        """Not implemented because tfp.distribution does not allow
           multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, inputs=None):
        """Depends on the distribution family of the observation.
        """
        raise NotImplementedError

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
        keys = jr.split(key, num_timesteps - 1)
        _, (states, emissions) = lax.scan(_step, initial_state, (keys, inputs[1:]))

        # Concatenate the initial state and emission with the following ones
        samp_states = jnp.concatenate((jnp.expand_dims(initial_state, 0), states))
        samp_emissions = jnp.concatenate((jnp.expand_dims(initial_emission, 0), emissions))
        return samp_states, samp_emissions

    def marginal_log_prob(self, emissions, inputs=None, params=None):
        """Compute log marginal likelihood of observations."""
        if params is None:
            # Compute marginal log prob using current parameter
            ssm_params = self._to_ssm_params(self.params)
        else:
            # Compute marginal log prob using given parameter
            ssm_params = self._to_ssm_params(params)

        filtered_posterior = self._ssm_filter(params=ssm_params, emissions=emissions, inputs=inputs)
        return filtered_posterior.marginal_loglik

    def posterior_sample(self, key, observed_time_series, inputs=None):
        num_timesteps, dim_obs = observed_time_series.shape
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        ssm_params = self._to_ssm_params(self.params)
        ll, states = self._ssm_posterior_sample(key, ssm_params, observed_time_series, inputs)
        obs_means = states @ self.emission_matrix.T + inputs @ self.params['regression_weights'].T
        obs_means = self._emission_constrainer(obs_means)
        obs = self.emission_distribution.sample(seed=key, sample_shape=num_timesteps)
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
        # obs_cov = self.params['emission_covariance']
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
            # marginal_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T + obs_cov
            # obs = self.emission_matrix @ prev_state + weights @ forecast_input\
            #     + MVN(jnp.zeros(dim_obs), obs_cov).sample(seed=key2)

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

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        raise NotImplementedError

    def _ssm_filter(self, params):
        """The filter of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_smoother(self, params):
        """The smoother of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        raise NotImplementedError

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space."""
        raise NotImplementedError


class GaussianSSM(_StructuralTimeSeriesSSM):

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
                 observation_regression_weights_prior=None):

        super().__init__(component_transition_matrices, component_observation_matrices,
                         component_initial_state_priors, component_transition_covariances,
                         component_transition_covariance_priors, cov_spars_matrices,
                         observation_regression_weights, observation_regression_weights_prior)
        # Add parameters of the observation covariance matrix.
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
        self.params.update({'emission_covariance': observation_covariance})
        self.param_props.update({'emission_covariance': emission_covariance_props})
        self.priors.update({'emission_covariance': observation_covariance_prior})

    def log_prior(self, params):
        # Compute sum of log priors of convariance matrices of the latent dynamics components,
        # as well as the log prior of parameters of the regression model (if the model has one).
        lp = super().log_prior(params)
        # Add log prior of covariance matrix of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        return lp

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        return MVN(self.emission_matrix @ state + self.params['regression_weights'] @ inputs,
                   self.params['emission_covariance'])

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


class PoissonSSM(_StructuralTimeSeriesSSM):
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
                 cov_spars_matrices,
                 observation_regression_weights=None,
                 observation_regression_weights_prior=None):

        super().__init__(component_transition_matrices, component_observation_matrices,
                         component_initial_state_priors, component_transition_covariances,
                         component_transition_covariance_priors, cov_spars_matrices,
                         observation_regression_weights, observation_regression_weights_prior)

    def emission_distribution(self, state, inputs=None):
        if inputs is None:
            inputs = jnp.array([0.])
        # Use the exponential function transform the unconstrained rate
        # to rate of the Poisson distribution
        return Poisson(
            log_rate=self.emission_matrix @ state + self.params['regression_weights'] @ inputs)

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


#iterated_conditional_moments_gaussian_filter(params, emissions, num_iter=2, inputs=None)
# cmgf_ekf_params = EKFParams(
#     initial_mean = jnp.zeros(state_dim),
#     initial_covariance = jnp.eye(state_dim),
#     dynamics_function = lambda z: random_rotation(state_dim, theta=jnp.pi/20) @ z,
#     dynamics_covariance = 0.001 * jnp.eye(state_dim),
#     emission_mean_function = lambda z: jnp.exp(poisson_weights @ z),
#     emission_var_function = lambda z: jnp.diag(jnp.exp(poisson_weights @ z))
# )