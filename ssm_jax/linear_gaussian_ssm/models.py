from jax import numpy as jnp
from jax import random as jr
from jax import lax
from jax.tree_util import register_pytree_node_class

from distrax import MultivariateNormalFullCovariance as MVN

from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, LGSSMParams
from ssm_jax.utils import PSDToRealBijector


_get_shape = lambda x, dim: x.shape[1:] if x.ndim == dim + 1 else x.shape

@register_pytree_node_class
class LinearGaussianSSM:
    """
    Linear Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
    p(y_t | z_t) = N(y_t | H_t z_t + D_t u_t + d_t, R_t)
    p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})
    where z_t = hidden, y_t = observed, u_t = inputs,
    dynamics_matrix = F
    dynamics_covariance = Q
    emission_matrix = H
    emissions_covariance = R
    initial_mean = mu_{1|0}
    initial_covariance = Sigma_{1|0}
    Optional parameters (default to 0)
    dynamics_input_matrix = B
    dynamics_bias = b
    emission_input_matrix = D
    emission_bias = d
    """

    def __init__(
        self,
        dynamics_matrix,
        dynamics_covariance,
        emission_matrix,
        emission_covariance,
        initial_mean=None,
        initial_covariance=None,
        dynamics_input_weights=None,
        dynamics_bias=None,
        emission_input_weights=None,
        emission_bias=None,
    ):
        self.emission_dim, self.state_dim = _get_shape(emission_matrix, 2)
        dynamics_input_dim = dynamics_input_weights.shape[1] if dynamics_input_weights is not None else 0
        emission_input_dim = emission_input_weights.shape[1] if emission_input_weights is not None else 0
        self.input_dim = max(dynamics_input_dim, emission_input_dim)

        # Save required args
        self.dynamics_matrix = dynamics_matrix
        self.dynamics_covariance = dynamics_covariance
        self.emission_matrix = emission_matrix
        self.emission_covariance = emission_covariance

        # Initialize optional args
        default = lambda x, v: x if x is not None else v
        self.initial_mean = default(initial_mean, jnp.zeros(self.state_dim))
        self.initial_covariance = default(initial_covariance, jnp.eye(self.state_dim))
        self.dynamics_input_weights = default(dynamics_input_weights, jnp.zeros((self.state_dim, self.input_dim)))
        self.dynamics_bias = default(dynamics_bias, jnp.zeros(self.state_dim))
        self.emission_input_weights = default(emission_input_weights, jnp.zeros((self.emission_dim, self.input_dim)))
        self.emission_bias = default(emission_bias, jnp.zeros(self.emission_dim))
        
        # Check shapes
        assert self.initial_mean.shape == (self.state_dim,)
        assert self.initial_covariance.shape == (self.state_dim, self.state_dim)
        assert self.dynamics_matrix.shape[-2:] == (self.state_dim, self.state_dim)
        assert self.dynamics_input_weights.shape[-2:] == (self.state_dim, self.input_dim)
        assert self.dynamics_bias.shape[-1:] == (self.state_dim,)
        assert self.dynamics_covariance.shape == (self.state_dim, self.state_dim)
        assert self.emission_input_weights.shape[-2:] == (self.emission_dim, self.input_dim)
        assert self.emission_bias.shape[-1:] == (self.emission_dim,)
        assert self.emission_covariance.shape == (self.emission_dim, self.emission_dim)

    @classmethod
    def random_initialization(cls, key, state_dim, emission_dim, input_dim=0):
        m1 = jnp.zeros(state_dim)
        Q1 = jnp.eye(state_dim)
        # TODO: Sample a random rotation matrix
        F = 0.99 * jnp.eye(state_dim)
        B = jnp.zeros((state_dim, input_dim))
        Q = 0.1 * jnp.eye(state_dim)
        H = jr.normal(key, (emission_dim, state_dim))
        D = jnp.zeros((emission_dim, input_dim))
        R = 0.1 * jnp.eye(emission_dim)
        return cls(
            dynamics_matrix=F,
            dynamics_covariance=Q,
            emission_matrix=H,
            emission_covariance=R,
            initial_mean=m1,
            initial_covariance=Q1,
            dynamics_input_weights=B,
            emission_input_weights=D
        )

    def sample(self, key, num_timesteps, inputs=None):
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Shorthand for parameters
        F = self.dynamics_matrix
        B = self.dynamics_input_weights
        b = self.dynamics_bias
        Q = self.dynamics_covariance
        H = self.emission_matrix
        D = self.emission_input_weights
        d = self.emission_bias
        R = self.emission_covariance

        def _step(carry, key_and_input):
            state = carry
            key, u = key_and_input

            # Sample data and next state
            key1, key2 = jr.split(key, 2)
            emission = MVN(H @ state + D @ u + d, R).sample(seed=key1)
            next_state = MVN(F @ state + B @ u + b, Q).sample(seed=key2)
            return next_state, (state, emission)

        # Initialize
        key, this_key = jr.split(key, 2)
        init_state = MVN(self.initial_mean, self.initial_covariance).sample(seed=this_key)

        # Run the sampler
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, init_state, (keys, inputs_dynamics, inputs_emission))
        return states, emissions

    def log_prob(self, states, emissions, inputs=None):
        num_timesteps = len(states)

        # Check shapes
        assert states.shape == (num_timesteps, self.state_dim)
        assert emissions.shape == (num_timesteps, self.emission_dim)
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        assert inputs.shape == (num_timesteps, self.input_dim)

        # Compute log prob
        lp = MVN(self.initial_mean, self.initial_covariance).log_prob(states[0])
        lp += (
            MVN(
                states[:-1] @ self.dynamics_matrix.T + inputs[:-1] @ self.dynamics_input_weights.T + self.dynamics_bias,
                self.dynamics_covariance,
            )
            .log_prob(states[1:])
            .sum()
        )
        lp += (
            MVN(
                states @ self.emission_matrix.T + inputs @ self.emission_input_weights.T + self.emission_bias,
                self.emission_covariance,
            )
            .log_prob(emissions)
            .sum()
        )
        return lp

    def marginal_log_prob(self, emissions, inputs=None):
        filtered_posterior = lgssm_filter(self, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        return lgssm_filter(self, emissions, inputs)

    def smoother(self, emissions, inputs=None):
        return lgssm_smoother(self, emissions, inputs)

    # Properties to allow unconstrained optimization and JAX jitting
    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            self.initial_mean,
            PSDToRealBijector.forward(self.initial_covariance),
            self.dynamics_matrix,
            self.dynamics_input_weights,
            self.dynamics_bias,
            PSDToRealBijector.forward(self.dynamics_covariance),
            self.emission_matrix,
            self.emission_input_weights,
            self.emission_bias,
            PSDToRealBijector.forward(self.emission_covariance),
        )

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_mean = unconstrained_params[0]
        initial_covariance = PSDToRealBijector.inverse(unconstrained_params[1])
        dynamics_matrix = unconstrained_params[2]
        dynamics_input_weights = unconstrained_params[3]
        dynamics_bias = unconstrained_params[4]
        dynamics_covariance = PSDToRealBijector.inverse(unconstrained_params[5])
        emission_matrix = unconstrained_params[6]
        emission_input_weights = unconstrained_params[7]
        emission_bias = unconstrained_params[8]
        emission_covariance = PSDToRealBijector.inverse(unconstrained_params[9])
        return cls(
            dynamics_matrix=dynamics_matrix,
            dynamics_covariance=dynamics_covariance,
            emission_matrix=emission_matrix,
            emission_covariance=emission_covariance,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_input_weights=dynamics_input_weights,
            dynamics_bias=dynamics_bias,
            emission_input_weights=emission_input_weights,
            emission_bias=emission_bias
        )

    @property
    def hyperparams(self):
        """Helper property to get a PyTree of model hyperparameters."""
        return tuple()

    # Use the to/from unconstrained properties to implement JAX tree_flatten/unflatten
    def tree_flatten(self):
        children = self.unconstrained_params
        aux_data = self.hyperparams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_unconstrained_params(children, aux_data)
