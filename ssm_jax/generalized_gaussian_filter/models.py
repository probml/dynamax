from jax import numpy as jnp
from jax import random as jr
from jax import lax, vmap
from jax.tree_util import register_pytree_node_class, tree_map
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from ssm_jax.generalized_gaussian_filter.inference import general_gaussian_filter


@register_pytree_node_class
class GeneralGaussianSSM:
    """
    General Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)
    p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)
    p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})
    where z_t = hidden, y_t = observed, u_t = inputs,
    dynamics_function = f
    dynamics_covariance = Q
    emission_function = h
    emissions_covariance = R
    initial_mean = mu_{1|0}
    initial_covariance = Sigma_{1|0}
    Optional parameters (default to 0)
    """

    def __init__(
        self,
        dynamics_function,
        dynamics_covariance,
        emission_function,
        emission_covariance,
        gaussian_expectation,
        gaussian_cross_covariance,
        initial_mean=None,
        initial_covariance=None,
    ):
        self.state_dim = dynamics_covariance.shape[0]
        self.emission_dim = emission_covariance.shape[0]

        # Save required args
        self.dynamics_function = dynamics_function
        self.dynamics_covariance = dynamics_covariance
        self.emission_function = emission_function
        self.emission_covariance = emission_covariance
        self.gaussian_expectation = gaussian_expectation
        self.gaussian_cross_covariance = gaussian_cross_covariance

        # Initialize optional args
        default = lambda x, v: x if x is not None else v
        self.initial_mean = default(initial_mean, jnp.zeros(self.state_dim))
        self.initial_covariance = default(initial_covariance, jnp.eye(self.state_dim))

        # Check shapes
        assert self.initial_mean.shape == (self.state_dim,)
        assert self.initial_covariance.shape == (self.state_dim, self.state_dim)
        assert self.dynamics_covariance.shape == (self.state_dim, self.state_dim)
        assert self.emission_covariance.shape == (self.emission_dim, self.emission_dim)

    def sample(self, key, num_timesteps, inputs=None):
        if isinstance(key, int):
            key = jr.PRNGKey(key)

        # Shorthand for parameters
        f = self.dynamics_function
        Q = self.dynamics_covariance
        h = self.emission_function
        R = self.emission_covariance

        if inputs is None:
            inputs = jnp.zeros((num_timesteps,))
            process_fn = lambda fn: (lambda x, u: fn(x))
            f, h = (process_fn(fn) for fn in (f, h))

        def _step(carry, key_and_input):
            state = carry
            key, u = key_and_input

            # Sample data and next state
            key1, key2 = jr.split(key, 2)
            emission = MVN(h(state, u), R).sample(seed=key1)
            next_state = MVN(f(state, u), Q).sample(seed=key2)
            return next_state, (state, emission)

        # Initialize
        key, subkey = jr.split(key, 2)
        init_state = MVN(self.initial_mean, self.initial_covariance).sample(seed=key)

        # Run the sampler
        keys = jr.split(subkey, num_timesteps)
        _, (states, emissions) = lax.scan(_step, init_state, (keys, inputs))
        return states, emissions

    def marginal_log_prob(self, emissions, inputs=None):
        filtered_posterior = general_gaussian_filter(self, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def ggf_filter(self, emissions, inputs=None):
        return general_gaussian_filter(self, emissions, inputs)

    # Properties to allow unconstrained optimization and JAX jitting
    @property
    def return_params(self):
        return (
            self.initial_mean,
            self.initial_covariance,
            self.dynamics_function,
            self.dynamics_covariance,
            self.emission_function,
            self.emission_covariance,
        )

    # Use the to/from unconstrained properties to implement JAX tree_flatten/unflatten
    def tree_flatten(self):
        return self.return_params()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)