import jax.numpy as jnp
import jax.scipy as jsp
from distrax import MultivariateNormalFullCovariance as MVN
from ssm_jax.abstractions import SSM
from ssm_jax.abstractions import Parameter
from ssm_jax.linear_gaussian_ssm.inference import LGSSMParams
from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter
from ssm_jax.utils import PSDToRealBijector


class LinearGaussianSTS(SSM):
    """Formulate the structual time series(STS) model into a LinearGaussianSSM model,
    which always have block-diagonal dynamics covariance matrix and fixed transition matrices.
    """

    def __init__(self,
                 component_transition_matrices,
                 component_observation_matrices,
                 component_initial_state_priors,
                 component_transition_covariance_priors,
                 observation_covariance,
                 observation_covariance_prior,
                 observation_regression_weights_prior=None):

        # Set parameters of the MultivariateGaussian distribution
        # for the initial state of the LinearGaussianSSM model
        self.initial_mean = jnp.array([
            initial_state_prior.mode() for initial_state_prior in component_initial_state_priors
        ]).flatten()

        self.initial_covariance = jsp.linalg.block_diag(*[
            initial_state_prior.covariance()
            for initial_state_prior in component_initial_state_priors
        ])

        # Set parameters of the dynamics model of the LinearGaussainSSM model
        self.dynamics_matrix = jsp.linalg.block_diag(*component_transition_matrices)
        self.state_dim = self.dynamics_matrix.shape[-1]
        self.dynamics_input_weights = jnp.zeros((self.state_dim, 0))
        self.dynamics_bias = jnp.zeros((self.state_dim, 1))

        # Set parameters of the emission model of the LinearGaussianSSM model
        self.emission_matrix = jnp.concatenate(tuple(component_observation_matrices), axis=1)
        self.emission_dim = self.emission_matrix.shape[-1]
        if observation_regression_weights_prior is not None:
            self.emission_input_weights = Parameter(observation_regression_weights_prior.mode())
            self.emission_input_weights_prior = observation_regression_weights_prior
        else:
            self.emission_input_weights = jnp.zeros((self.emission_dim, 0))
        self.emission_bias = jnp.zeros((self.emission_dim, 1))
        self.emission_covariance = Parameter(observation_covariance, bijector=PSDToRealBijector)
        self.emission_covariance_prior = observation_covariance_prior

        # Set dynamics covariance matrix of the LinearGaussianSSM model
        self.num_blocks = len(component_transition_covariance_priors)
        assert self.state_dim == self.num_blocks * self.emission_dim

        spars_matrix = jnp.zeros((self.state_dim, 0))
        self.dynamics_covariance_set = {}
        for i in self.num_blocks:
            c_prior = component_transition_covariance_priors[i]
            spars_block = jnp.zeros((self.state_dim, self.emission_dim))
            if c_prior is not None:
                setattr(self, f'dynamics_cov_block_{i}',
                        Parameter(c_prior.mode(), bijector=PSDToRealBijector))
                self.dynamics_covariance_set[f'dynamics_cov_block_{i}'] = c_prior
                spars_block = spars_block.at[i * self.emission_dim:(i + 1) * self.emission_dim -
                                             1, :].set(jnp.eye(self.emission_dim))
            spars_matrix = jnp.concatenate((spars_matrix, spars_block), axis=1)

        self.dynamics_covariance_dense = jsp.linalg.block_diag(self.dynamics_covariance_set)
        self.dynamics_covariance = spars_matrix @ self.dynamics_covariance_dense @ spars_matrix.T

    # Set component distributions of SSM
    def initial_distribution(self, **covariates):
        return MVN(self.initial_mean, self.initial_covariance)

    def transition_distribution(self, state, **covariates):
        return MVN(self.dynamics_matrix @ state, self.dynamics_covariance)

    def emission_distribution(self, state, **covariates):
        input = covariates['inputs'] if 'inputs' in covariates else jnp.zeros(self.input_dim)
        return MVN(self.emission_matrix @ state + self.emission_input_weights @ input,
                   self.emission_covariance)

    def log_prior(self):
        lp = jnp.array([
            self.dynamics_covariance_set[block].log_prob(getattr(self, block).value)
            for block in self.dynamics_covariance_set.keys
        ]).sum()
        # log prior of the emission model
        lp += self.emission_covariance_prior.log_prob(self.emission_covariance.value)
        if isinstance(self.emission_input_weights, Parameter):
            lp += self.emission_input_weights_prior.log_prob(self.emission_input_weights.value)

        return lp

    @property
    def _lgssm_params(self):
        return LGSSMParams(initial_mean=self.initial_mean,
                           initial_covariance=self.initial_covariance,
                           dynamics_matrix=self.dynamics_matrix,
                           dynamics_input_weights=self.dynamics_input_weights,
                           dynamics_bias=self.dynamics_bias,
                           dynamics_covariance=self.dynamics_covariance,
                           emission_matrix=self.emission_matrix,
                           emission_input_weights=self.emission_input_weights,
                           emission_bias=self.emission_bias,
                           emission_covariance=self.emission_covariance)

    def marginal_log_prob(self, emissions):
        posterior = lgssm_filter(self._lgssm_params, emissions)
        return posterior.marginal_loglik
