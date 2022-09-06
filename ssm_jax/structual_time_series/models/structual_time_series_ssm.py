import blackjax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from distrax import MultivariateNormalFullCovariance as MVN
from jax import jit
from jax import vmap
from ssm_jax.abstractions import SSM
from ssm_jax.abstractions import Parameter
from ssm_jax.linear_gaussian_ssm.inference import LGSSMParams
from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter
from ssm_jax.utils import PSDToRealBijector
from tqdm.auto import trange


class StructualTimeSeriesSSM(SSM):
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

        # Set parameters for the initial state of the LinearGaussianSSM model
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
        self.emission_dim = self.emission_matrix.shape[0]
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

        self.dynamics_covariance_set = {}
        self.dynamics_covariance_prior_set = {}
        sparse_matrix = jnp.zeros((self.state_dim, 0))
        for i in range(self.num_blocks):
            c_prior = component_transition_covariance_priors[i]
            sparse_block = jnp.zeros((self.state_dim, self.emission_dim))
            if c_prior is not None:
                setattr(self, f'dynamics_cov_block_{i}',
                        Parameter(c_prior.mode(), bijector=PSDToRealBijector))
                self.dynamics_covariance_set[f'dynamics_cov_block_{i}'] = c_prior.mode()
                self.dynamics_covariance_prior_set[f'dynamics_cov_block_{i}'] = c_prior
                sparse_block = sparse_block.at[i * self.emission_dim:(i + 1) * self.emission_dim -
                                               1, :].set(jnp.eye(self.emission_dim))
                sparse_matrix = jnp.concatenate((sparse_matrix, sparse_block), axis=1)

        self.dynamics_covariance_dense = jsp.linalg.block_diag(
            *self.dynamics_covariance_set.values())
        self.dynamics_covariance = sparse_matrix @ self.dynamics_covariance_dense @ sparse_matrix.T

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
            self.dynamics_covariance_prior_set[block].log_prob(getattr(self, block).value)
            for block in self.dynamics_covariance_set.keys()
        ]).sum()
        # log prior of the emission model
        lp += self.emission_covariance_prior.log_prob(self.emission_covariance.value)
        if isinstance(self.emission_input_weights, Parameter):
            lp += self.emission_input_weights_prior.log_prob(self.emission_input_weights.value)

        return lp

    @property
    def lgssm_params(self):
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
        posterior = lgssm_filter(self.lgssm_params, emissions)
        return posterior.marginal_loglik

    def fit_hmc(self,
                key,
                sample_size,
                batch_emissions,
                batch_inputs=None,
                warmup_steps=500,
                num_integration_steps=30):
        """Sample parameters of the model using HMC."""

        # The log likelihood that the HMC samples from
        def _logprob(hmc_position, batch_emissions=batch_emissions, batch_inputs=batch_inputs):
            self.unconstrained_params = hmc_position
            batch_lls = vmap(self.marginal_log_prob)(batch_emissions, batch_inputs)
            lp = self.log_prior() + batch_lls.sum()
            return lp

        # Initialize the HMC sampler using window_adaptation
        hmc_initial_position = self.unconstrained_params
        warmup = blackjax.window_adaptation(blackjax.hmc,
                                            _logprob,
                                            num_steps=warmup_steps,
                                            num_integration_steps=num_integration_steps)
        hmc_initial_state, hmc_kernel, _ = warmup.run(key, hmc_initial_position)
        hmc_kernel = jit(hmc_kernel)

        @jit
        def one_step(current_state, rng_key):
            next_state, _ = hmc_kernel(rng_key, current_state)
            return next_state

        # Start sampling
        keys = iter(jr.split(key, sample_size))
        param_samples = []
        current_state = hmc_initial_state
        for _ in trange(sample_size):
            current_state = one_step(current_state, next(keys))
            param_samples.append(current_state.position)

        return self._to_complete_parameters(param_samples)

    def _to_complete_parameters(self, unconstrained_params):
        items = sorted(self.__dict__.items())
        names = [key for key, prm in items if isinstance(prm, Parameter) and not prm.is_frozen]

        @jit
        def join_params(unconstrained_parameter):
            current_params = self.params
            for i in range(len(unconstrained_parameter)):
                value = getattr(self, names[i]).bijector.inverse(unconstrained_parameter[i])
                setattr(current_params, names[i][1:], value)
            return current_params

        return [join_params(param) for param in unconstrained_params]
