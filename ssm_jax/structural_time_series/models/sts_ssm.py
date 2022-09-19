import blackjax
from collections import OrderedDict
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax import jit
from jax import vmap, lax
from ssm_jax.abstractions import SSM
from ssm_jax.linear_gaussian_ssm.inference import LGSSMParams
from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter
from ssm_jax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tqdm.auto import trange
from ssm_jax.structural_time_series.new_parameters import (
    to_unconstrained, from_unconstrained, ParameterProperties)
import tensorflow_probability as tfp
tfb = tfp.bijectors
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map


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
        dynamics_unc_cov_priors = OrderedDict()
        for c in component_transition_covariance_priors.keys():
            dynamics_covariance_props[c] = ParameterProperties(
                trainable=True, constrainer=tfb.Invert(PSDToRealBijector))
            dynamics_unc_cov_priors[c] = tfd.TransformedDistribution(
                distribution=component_transition_covariance_priors[c],
                bijector=PSDToRealBijector)
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
        emission_unc_cov_prior = tfd.TransformedDistribution(
            distribution=observation_covariance_prior, bijector=PSDToRealBijector)

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

        self.unc_priors = {'dynamics_covariances': dynamics_unc_cov_priors,
                           'emission_covariance': emission_unc_cov_prior,
                           'input_weights': emission_input_weights_prior}

    def log_prior(self):
        lp = jnp.array([cov_prior.log_prob(cov) for cov, cov_prior in
                        zip(self.params['dynamics_covariances'].values(),
                            self.priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.priors['emission_covariance'].log_prob(self.params['emission_covariance'])
        if self.params['input_weights']:
            lp += self.priors['input_weights'].log_prob(self.params['input_weights'])
        return lp

    def log_unc_prior(self, unc_params):
        lp = jnp.array([unc_cov_prior.log_prob(unc_cov) for unc_cov, unc_cov_prior in
                        zip(unc_params['dynamics_covariances'].values(),
                            self.unc_priors['dynamics_covariances'].values())]).sum()
        # log prior of the emission model
        lp += self.unc_priors['emission_covariance'].log_prob(unc_params['emission_covariance'])
        if unc_params['input_weights']:
            lp += self.unc_priors['input_weights'].log_prob(unc_params['input_weights'])
        return lp

    def to_lgssm_params(self, params):
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
            lgssm_params = self.to_lgssm_params(self.params)
        else:
            # Compute marginal log prob using given parameter
            lgssm_params = self.to_lgssm_params(params)

        filtered_posterior = lgssm_filter(lgssm_params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def fit_hmc(self,
                key,
                sample_size,
                batch_emissions,
                batch_inputs=None,
                warmup_steps=500,
                num_integration_steps=30):
        unc_params = self.unc_params

        def logprob(trainable_unc_params):
            unc_params.update(trainable_unc_params)
            log_pri = self.log_unc_prior(unc_params)
            params = from_unconstrained(trainable_unc_params, fixed_params, self.param_props)
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
            sample = from_unconstrained(unc_sample, fixed_params, self.param_props)
            param_samples.append(sample)

        return param_samples

    @property
    def unc_params(self):
        unc_ems_cov = self.param_props['emission_covariance'].constrainer.inverse(
            self.params['emission_covariance'])
        unc_dyn_cov = OrderedDict()
        for k, v in self.params['dynamics_covariances'].items():
            unc_dyn_cov[k] = self.param_props['dynamics_covariances'][k].constrainer.inverse(v)
        unc_params = {'dynamics_covariances': unc_dyn_cov,
                      'emission_covariance': unc_ems_cov,
                      'input_weights': self.params['input_weights']}
        return unc_params

    # Set component distributions of SSM
    def initial_distribution(self, **covariates):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state, **covariates):
        covariance = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        dynamics_covariance = spars_matrix @ covariance @ spars_matrix.T
        return MVN(self.dynamics_matrix @ state, dynamics_covariance)

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
        initial_emission = self.emission_distribution(initial_state, **initial_covariate).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_covariates = tree_map(lambda x: x[1:], covariates)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_covariates))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def filter(self, emissions, inputs=None):
        sts_ssm = self.as_ssm()
        states = sts_ssm.filter(emissions, inputs)
        component_states = self._split_joint_states(states)
        return component_states

    # def forecast(self,
    #              key,
    #              observed_time_series,
    #              num_forecast_steps,
    #              inputs=None):
    #     lgssm_params = self.to_lgssm_params(self.params)
    #     filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, inputs)
    #     filtered_mean = filtered_posterior.filtered_means
    #     filtered_cov = filtered_posterior.filtered_covariances

    #     covariance = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
    #     _d = covariance.shape[0]
    #     spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())

    #     initial_distribution = MVN(self.dynamics_matrix @ filtered_mean[-1],
    #                                self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T
    #                                + spars_matrix @ covariance @ spars_matrix.T)

    #     def _step(prev_state, key):
    #         key1, key2 = jr.split(key, 2)
    #         # state = self.transition_distribution(prev_state, **covariate).sample(seed=key2)
    #         state = self.dynamics_matrix @ prev_state\
    #             + spars_matrix @ MVN(jnp.zeros(_d), covariance).sample(seed=key2)
    #         emission = self.emission_distribution(state).sample(seed=key1)
    #         return state, (state, emission)

    #     # Sample the initial state
    #     key1, key2, key = jr.split(key, 3)
    #     initial_state = initial_distribution.sample(seed=key1)
    #     initial_emission = self.emission_distribution(initial_state).sample(seed=key2)

    #     # Sample the remaining emissions and states
    #     next_keys = jr.split(key, num_forecast_steps - 1)
    #     _, (next_states, next_emissions) = lax.scan(_step, initial_state, next_keys)

    #     # Concatenate the initial state and emission with the following ones
    #     expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    #     states = tree_map(expand_and_cat, initial_state, next_states)
    #     emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
    #     return emissions

    def forecast(self,
                 key,
                 observed_time_series,
                 num_forecast_steps,
                 inputs=None):
        lgssm_params = self.to_lgssm_params(self.params)
        filtered_posterior = lgssm_filter(lgssm_params, observed_time_series, inputs)
        filtered_mean = filtered_posterior.filtered_means
        filtered_cov = filtered_posterior.filtered_covariances

        covariance = jsp.linalg.block_diag(*self.params['dynamics_covariances'].values())
        spars_matrix = jsp.linalg.block_diag(*self.spars_matrix.values())
        cov = spars_matrix @ covariance @ spars_matrix.T

        initial_mean = self.dynamics_matrix @ filtered_mean[-1]
        initial_cov = self.dynamics_matrix @ filtered_cov[-1] @ self.dynamics_matrix.T + cov

        def _step(prev_params, _=None):
            prev_mean, prev_cov = prev_params
            next_mean = self.dynamics_matrix @ prev_mean
            next_cov = self.dynamics_matrix @ prev_cov @ self.dynamics_matrix.T + cov
            marginal_cov = self.emission_matrix @ next_cov @ self.emission_matrix.T\
                + self.params['emission_covariance']
            return (next_mean, next_cov), (self.emission_matrix @ next_mean, marginal_cov)

        # Sample the initial state
        initial_params = (initial_mean, initial_cov)
        _, (next_mean, next_cov) = lax.scan(_step, initial_params, None, length=num_forecast_steps)

        return next_mean, next_cov
