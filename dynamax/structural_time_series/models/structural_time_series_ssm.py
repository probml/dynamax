import blackjax
from collections import OrderedDict
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from jaxopt import LBFGS
from dynamax.abstractions import SSM
from dynamax.cond_moments_gaussian_filter.cmgf import (
    iterated_conditional_moments_gaussian_filter as cmgf_filt,
    iterated_conditional_moments_gaussian_smoother as cmgf_smooth,
    EKFIntegrals)
from dynamax.cond_moments_gaussian_filter.generalized_gaussian_ssm import GGSSMParams
from dynamax.linear_gaussian_ssm.inference import (
    LGSSMParams,
    lgssm_filter,
    lgssm_smoother,
    lgssm_posterior_sample)
from dynamax.parameters import (
    to_unconstrained,
    from_unconstrained,
    log_det_jac_constrain,
    flatten,
    unflatten,
    ParameterProperties)
from dynamax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN,
    MultivariateNormalDiag as MVNDiag,
    Poisson)
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
        self.component_emission_matrices = component_observation_matrices
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

    @property
    def emission_shape(self):
        return (self.emission_dim,)

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
        obs = self.emission_distribution(states, inputs).sample(seed=key,
                                                                sample_shape=num_timesteps)
        return obs_means, obs

    def component_posterior(self, emissions, inputs):
        """Smoothing by component
        """
        # Compute the posterior of the joint SSM model
        component_pos = OrderedDict()
        ssm_params = self._to_ssm_params(self.params)
        pos = self._ssm_smoother(ssm_params, emissions, inputs)
        mu_pos = pos.smoothed_means
        var_pos = pos.smoothed_covariances

        # Decompose by component
        _loc = 0
        for c, emission_matrix in self.component_emission_matrices.items():
            c_dim = emission_matrix.shape[-1]
            c_mu = mu_pos[:, _loc:_loc+c_dim]
            c_var = var_pos[:, _loc:_loc+c_dim, _loc:_loc+c_dim]
            c_emission_mu = vmap(lambda s, m: m @ s, (0, None))(c_mu, emission_matrix)
            c_emission_constrained_mu = self._emission_constrainer(c_emission_mu)
            c_emission_var = vmap(lambda s, m: m @ s @ m.T, (0, None))(c_var, emission_matrix)
            component_pos[c] = (c_emission_constrained_mu, c_emission_var)
            _loc += c_dim

        # Include the regression effect if the model has the regression component
        if inputs is not None:
            W = self.params['regression_weights']
            regression_effect = vmap(lambda w, x: w @ x, (None, 0))(W, inputs)
            # Given the regression weight, the regression effect is not random
            component_pos['Regression'] = (regression_effect, jnp.zeros((inputs.shape[0], W.shape[0])))

        return component_pos

    def fit_hmc(self,
                key,
                sample_size,
                emissions,
                inputs=None,
                warmup_steps=500,
                num_integration_steps=20):

        def unnorm_log_pos(trainable_unc_params):
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptations
        hmc_initial_position, fixed_params = to_unconstrained(self.params, self.param_props)
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            unnorm_log_pos,
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

    def fit_vi(self, key, sample_size, emissions, inputs=None, M=100):
        """
        ADVI approximate the posterior distribtuion p of unconstrained global parameters
        with factorized multivatriate normal distribution:
        q = \prod_{k=1}^{K} q_k(mu_k, sigma_k),
        where K is dimension of p.

        The hyper-parameters of q to be optimized over are (mu_k, log_sigma_k))_{k=1}^{K}.

        The trick of reparameterization is employed to reduce the variance of SGD,
        which is achieved by written KL(q || p) as expectation over standard normal distribution
        so a sample from q is obstained by
        s = z * exp(log_sigma_k) + mu_k,
        where z is a sample from the standard multivarate normal distribtion.

        This implementation of ADVI uses fixed samples from q during fitting, instead of
        updating samples from q in each iteration, as in SGD.
        So the second order fixed optimization algorithm L-BFGS is used.

        Args:
            sample_size (int): number of samples to be returned from the fitted approxiamtion q.
            M (int): number of fixed samples from q used in evaluation of ELBO.

        Returns:
            Samples from the approximate posterior q.
        """
        key0, key1 = jr.split(key)
        model_unc_params, fixed_params = to_unconstrained(self.params, self.param_props)
        params_flat, params_structure = flatten(model_unc_params)
        vi_dim = len(params_flat)

        std_normal = MVNDiag(jnp.zeros(vi_dim), jnp.ones(vi_dim))
        std_samples = std_normal.sample(seed=key0, sample_shape=(M,))
        std_samples = vmap(unflatten, (None, 0))(params_structure, std_samples)

        @jit
        def unnorm_log_pos(unc_params):
            """Unnormalzied log posterior of global parameters."""

            params = from_unconstrained(unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        @jit
        def _samp_elbo(vi_params, std_samp):
            """Evaluate ELBO at one sample from the approximate distribution q.
            """
            vi_means, vi_log_sigmas = vi_params
            # unc_params_flat = vi_means + std_samp * jnp.exp(vi_log_sigmas)
            # unc_params = unflatten(params_structure, unc_params_flat)
            # With reparameterization, entropy of q evaluated at z is
            # sum(hyper_log_sigma) plus some constant depending only on z.
            _params = tree_map(lambda x, log_sig: x * jnp.exp(log_sig), std_samp, vi_log_sigmas)
            unc_params = tree_map(lambda x, mu: x + mu, _params, vi_means)
            q_entropy = flatten(vi_log_sigmas)[0].sum()
            return q_entropy + unnorm_log_pos(unc_params)

        objective = lambda x: -vmap(_samp_elbo, (None, 0))(x, std_samples).mean()

        # Fit ADVI with LBFGS algorithm
        initial_vi_means = model_unc_params
        initial_vi_log_sigmas = unflatten(params_structure, jnp.zeros(vi_dim))
        initial_vi_params = (initial_vi_means, initial_vi_log_sigmas)
        lbfgs = LBFGS(maxiter=1000, fun=objective, tol=1e-3, stepsize=1e-3, jit=True)
        (vi_means, vi_log_sigmas), _info = lbfgs.run(initial_vi_params)

        # Sample from the learned approximate posterior q
        _samples = std_normal.sample(seed=key1, sample_shape=(sample_size,))
        _vi_means = flatten(vi_means)[0]
        _vi_log_sigmas = flatten(vi_log_sigmas)[0]
        vi_samples_flat = _vi_means + _samples * jnp.exp(_vi_log_sigmas)
        vi_unc_samples = vmap(unflatten, (None, 0))(params_structure, vi_samples_flat)
        vi_samples = vmap(from_unconstrained, (0, None, None))(
            vi_unc_samples, fixed_params, self.param_props)

        return vi_samples

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        """Forecast the time series"""

        if forecast_inputs is None:
            forecast_inputs = jnp.zeros((num_forecast_steps, 0))
        weights = self.params['regression_weights']
        comp_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        dim_comp = comp_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        ssm_params = self._to_ssm_params(self.params)
        filtered_posterior = self._ssm_filter(params=ssm_params,
                                              emissions=observed_time_series,
                                              inputs=past_inputs)
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
            marginal_mean = self._emission_constrainer(marginal_mean)
            emission_mean_cov = self.emission_matrix @ prev_cov @ self.emission_matrix.T
            obs = self.emission_distribution(prev_state, forecast_input).sample(seed=key2)

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), comp_cov).sample(seed=key1)

            return (next_mean, next_cov, next_state), (marginal_mean, emission_mean_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov, initial_state)
        _, (ts_means, ts_mean_covs, ts) = lax.scan(_step, initial_params, (keys, forecast_inputs))

        return ts_means, ts_mean_covs, ts

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        raise NotImplementedError

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_smoother(self, params, emissions, inputs):
        """The smoother of the corresponding SSM model"""
        raise NotImplementedError

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        raise NotImplementedError

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space."""
        raise NotImplementedError


#####################################################################
# SSM classes for STS model with specific observation distributions #
#####################################################################


class GaussianSSM(_StructuralTimeSeriesSSM):
    """SSM classes for STS model where the observations follow multivariate normal distributions.
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

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs
            )
        ts_covs = ts_mean_covs + self.params['emission_covariance']
        return ts_means, ts_covs, ts

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
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

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return lgssm_filter(params=params, emissions=emissions, inputs=inputs)

    def _ssm_smoother(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return lgssm_smoother(params=params, emissions=emissions, inputs=inputs)

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        return lgssm_posterior_sample(rng=key,
                                      params=ssm_params,
                                      emissions=observed_time_series,
                                      inputs=inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
           Use identity transformation when the observation distribution is MVN.
        """
        return emission


class PoissonSSM(_StructuralTimeSeriesSSM):
    """SSM classes for STS model where the observations follow Poisson distributions.
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
        log_rate = self.emission_matrix @ state + self.params['regression_weights'] @ inputs
        return Poisson(rate=self._emission_constrainer(log_rate))

    def forecast(self, key, observed_time_series, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        ts_means, ts_mean_covs, ts = super().forecast(
            key, observed_time_series, num_forecast_steps, past_inputs, forecast_inputs
            )
        _sample = lambda r, key: Poisson(rate=r).sample(seed=key)
        ts_samples = vmap(_sample)(ts_means, jr.split(key, num_forecast_steps))
        return ts_samples, ts_means, ts

    def _to_ssm_params(self, params):
        """Wrap the STS model into the form of the corresponding SSM model """
        comp_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ comp_cov @ spars_matrix.T
        return GGSSMParams(initial_mean=self.initial_mean,
                         initial_covariance=self.initial_covariance,
                         dynamics_function=lambda z: self.dynamics_matrix @ z,
                         dynamics_covariance=spars_cov,
                         emission_mean_function=
                         lambda z: self._emission_constrainer(self.emission_matrix @ z),
                         emission_cov_function=
                         lambda z: jnp.diag(self._emission_constrainer(self.emission_matrix @ z)))

    def _ssm_filter(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_filt(params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_smoother(self, params, emissions, inputs):
        """The filter of the corresponding SSM model"""
        return cmgf_smooth(params=params, inf_params=EKFIntegrals(), emissions=emissions, inputs=inputs, num_iter=2)

    def _ssm_posterior_sample(self, key, ssm_params, observed_time_series, inputs):
        """The posterior sampler of the corresponding SSM model"""
        # TODO:
        # Implement the real posteriror sample.
        # Currently it simply returns the filtered means.
        print('Currently the posterior_sample for STS model with Poisson likelihood\
               simply returns the filtered means.')
        return self._ssm_filter(ssm_params, observed_time_series, inputs)

    def _emission_constrainer(self, emission):
        """Transform the state into the possibly constrained space.
        """
        # Use the exponential function to transform the unconstrained rate
        # to rate of the Poisson distribution
        return jnp.exp(emission)
