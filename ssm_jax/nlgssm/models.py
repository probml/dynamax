from functools import partial
from typing import Callable
from inspect import signature

from jax import numpy as jnp
from jax import random as jr
from jax import lax, vmap
from jax.tree_util import register_pytree_node_class, tree_map

from distrax import MultivariateNormalFullCovariance as MVN
import chex

from ssm_jax.nlgssm.extended_inference import extended_kalman_filter


@chex.dataclass
class NLGSSMParams:
    """Lightweight container for NLGSSM parameters.
    """
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_function: Callable
    emission_covariance: chex.Array

@chex.dataclass
class NLGSSMPosterior:
    """Simple wrapper for properties of an NLGSSM posterior distribution.

    Attributes:
            marginal_loglik: marginal log likelihood of the data
            filtered_means: (T,D_hid) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_covariances: (T,D_hid,D_hid) array,
                Cov[x_t | y_{1:t}, u_{1:t}].
            smoothed_means: (T,D_hid) array,
                E[x_t | y_{1:T}, u_{1:T}].
            smoothed_covs: (T,D_hid,D_hid) array of smoothed marginal covariances,
                Cov[x_t | y_{1:T}, u_{1:T}].
            smoothed_cross: (T-1, D_hid, D_hid) array of smoothed cross products,
                E[x_t x_{t+1}^T | y_{1:T}, u_{1:T}].
    """
    marginal_loglik: chex.Scalar = None
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None
    smoothed_cross_covariances: chex.Array = None

@register_pytree_node_class
class NonLinearGaussianSSM:
    '''
    Non-Linear Gaussian State Space Model is defined as follows:
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
    
    TODO: Add additional parameters for non-EKF algorithms 
    (e.g.) alpha, beta, kappa for UKF
    '''
    def __init__(self,
                 dynamics_function,
                 dynamics_covariance,
                 emission_function,
                 emission_covariance,
                 initial_mean=None,
                 initial_covariance=None):
        self.state_dim = dynamics_covariance.shape[0]
        self.emission_dim = emission_covariance.shape[0]

        # Save required args
        self.dynamics_function = dynamics_function
        self.dynamics_covariance = dynamics_covariance
        self.emission_function = emission_function
        self.emission_covariance = emission_covariance

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
        filtered_posterior = extended_kalman_filter(self, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        return extended_kalman_filter(self, emissions, inputs)