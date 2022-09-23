import blackjax
from collections import OrderedDict
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.tree_util import tree_map
from ssm_jax.abstractions import SSM
from ssm_jax.linear_gaussian_ssm.inference import (
    LGSSMParams, lgssm_filter, lgssm_smoother, lgssm_posterior_sample)
from ssm_jax.structural_time_series.new_parameters import (
    to_unconstrained, from_unconstrained, ParameterProperties)
from ssm_jax.utils import PSDToRealBijector
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tqdm.auto import trange

tfb = tfp.bijectors


class StructuralTimeSeriesSSM(SSM):
    """Formulate the structual time series(STS) model into a LinearGaussianSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    The covariance matrix of the dynamics model takes the form:
    R @ Q
    where Q is a dense matrix (blockwise diagonal),
    R is the sparsing matrix containing zeros and ones.
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

        # Set parameters for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.concatenate(
            [init_pri.mode() for init_pri in component_initial_state_priors.values()])
        self.initial_covariance = jsp.linalg.block_diag(
            *[init_pri.covariance() for init_pri in component_initial_state_priors.values()])

        # Set parameters of the dynamics model of the LinearGaussainSSM model
        self.dynamics_matrix = jsp.linalg.block_diag(*component_transition_matrices.values())
        self.state_dim = self.dynamics_matrix.shape[-1]
        self.dynamics_input_weights = jnp.zeros((self.state_dim, 0))
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
            emission_input_weights_props = ParameterProperties(
                trainable=True, constrainer=tfb.Identity)
            emission_input_weights_prior = observation_regression_weights_prior
        else:
            emission_input_weights = jnp.zeros((self.emission_dim, 0))
            emission_input_weights_props = ParameterProperties(
                trainable=False, constrainer=tfb.Identity)
            emission_input_weights_prior = None
        self.emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance_props = ParameterProperties(
            trainable=True, constrainer=tfb.Invert(PSDToRealBijector))

        # Parameters, their properties, and priors of the SSM model
        self.params = {'dynamics_covariances': component_transition_covariances,
                       'emission_covariance': observation_covariance,
                       'input_weights': emission_input_weights}

        self.param_props = {'dynamics_covariances': dynamics_covariance_props,
                            'emission_covariance': emission_covariance_props,
                            'input_weights': emission_input_weights_props}

        self.priors = {'dynamics_covariances': component_transition_covariance_priors,
                       'emission_covariance': observation_covariance_prior,
                       'input_weights': emission_input_weights_prior}

    def log_prior(self, params):
        lp = jnp.array([cov_prior.log_prob(cov) for cov, cov_prior in
                        zip(params['dynamics_covariances'].values(),
                            self.priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.priors['emission_covariance'].log_prob(params['emission_covariance'])
        if params['input_weights']:
            lp += self.priors['input_weights'].log_prob(params['input_weights'])
        return lp

    # Set component distributions of SSM
    def initial_distribution(self, **covariates):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state, **covariates):
        """Not implemented because tfp.distribution does not allow
        multivariate normal distribution with singular convariance matrix.
        """
        raise NotImplementedError

    def emission_distribution(self, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates and covariates['inputs'] is not None\
            else jnp.zeros(self.params['input_weights'].shape[-1])
        return MVN(self.emission_matrix @ state + self.params['input_weights'] @ input,
                   self.params['emission_covariance'])

    def sample(self, key, num_timesteps, **covariates):
        """Sample a sequence of latent states and emissions.
        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """
        covariance = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        _d = covariance.shape[0]
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())

        def _step(prev_state, args):
            key, covariate = args
            key1, key2 = jr.split(key, 2)
            # state = self.transition_distribution(prev_state, **covariate).sample(seed=key2)
            state = prev_state + spars_matrix @ MVN(jnp.zeros(_d), covariance).sample(seed=key2)
            emission = self.emission_distribution(state, **covariate).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_covariate = tree_map(lambda x: x[0], covariates)
        initial_state = self.initial_distribution(**initial_covariate).sample(seed=key1)
        initial_emission = self.emission_distribution(
            initial_state, **initial_covariate
            ).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_covariates = tree_map(lambda x: x[1:], covariates)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_covariates))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def _to_lgssm_params(self, params):
        dyn_cov = jsp.linalg.block_diag(*params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        dynamics_covariance = spars_matrix @ dyn_cov @ spars_matrix.T
        emission_covariance = params['emission_covariance']
        emission_input_weights = params['input_weights']
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=self.dynamics_matrix,
                           dynamics_input_weights=self.dynamics_input_weights,
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=dynamics_covariance,
                           emission_matrix=self.emission_matrix,
                           emission_input_weights=emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=emission_covariance)

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

    def posterior_sample(self, key, emissions=None):
        lgssm_params = self._to_lgssm_params(self.params)
        ll, states = lgssm_posterior_sample(key, lgssm_params, emissions, inputs=None)
        obs = states
        return obs

    def fit_hmc(self,
                key,
                sample_size,
                batch_emissions,
                batch_inputs=None,
                warmup_steps=500,
                num_integration_steps=30):

        def logprob(trainable_unc_params):
            params, log_det_jacobian = from_unconstrained(
                trainable_unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jacobian
            batch_lls = vmap(self.marginal_log_prob)(batch_emissions, batch_inputs, params)
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
            sample, _ = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        param_samples = tree_map(lambda x, *y: jnp.array([x] + [i for i in y]),
                                 param_samples[0], *param_samples[1:])
        return param_samples

    def forecast(self, key, observed_time_series, num_forecast_steps, inputs=None):
        """Forecast the time series"""

        dense_cov = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        emiss_cov = self.params['emission_covariance']
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        spars_cov = spars_matrix @ dense_cov @ spars_matrix.T
        dim_obs = observed_time_series.shape[-1]
        dim_comp = dense_cov.shape[-1]

        # Filtering the observed time series to initialize the forecast
        lgssm_params = self._to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + spars_cov

        # def _step(prev_params, _=None):
        #     prev_mean, prev_cov = prev_params
        #     next_mean = self.dynamics_matrix @ prev_mean
        #     next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + cov
        #     marginal_cov = self.emission_matrix @ next_cov @ self.emission_matrix.T\
        #         + self.params['emission_covariance']
        #     return (next_mean, next_cov), (self.emission_matrix @ next_mean, marginal_cov)
        def _step(prev_params, key):
            key1, key2 = jr.split(key)
            prev_mean, prev_cov, prev_state = prev_params

            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + spars_cov
            next_state = self.dynamics_matrix @ prev_state\
                + spars_matrix @ MVN(jnp.zeros(dim_comp), dense_cov).sample(seed=key1)

            marginal_mean = self.emission_matrix @ next_mean
            marginal_cov = self.emission_matrix @ next_cov @ self.emission_matrix.T + emiss_cov
            obs = self.emission_matrix @ next_state\
                + MVN(jnp.zeros(dim_obs), emiss_cov).sample(seed=key2)

            return (next_mean, next_cov, next_state), (marginal_mean, marginal_cov, obs)

        # Initialize
        keys = jr.split(key, num_forecast_steps)
        initial_params = (initial_mean, initial_cov)
        _, (ts_means, ts_covs, ts) = lax.scan(_step, initial_params, keys)

        return ts_means, ts_covs, ts
