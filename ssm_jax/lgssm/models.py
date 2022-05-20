from jax import numpy as jnp
from jax import random as jr
from jax import lax
from distrax import MultivariateNormalFullCovariance as MVN
import chex

@chex.dataclass
class LGSSMParams:
    """Lightweight container for LGSSM parameters."""
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_matrix: chex.Array
    dynamics_input_weights: chex.Array
    dynamics_covariance: chex.Array
    emission_matrix: chex.Array
    emission_input_weights: chex.Array
    emission_covariance: chex.Array
    is_stationary: chex.Array = True

    @property
    def state_dim(self):
        return self.emission_matrix.shape[-1]

    @property
    def emission_dim(self):
        return self.emission_matrix.shape[-2]


def lgssm_joint_sample(rng, params, num_timesteps, inputs):
    """_summary_

    Args:
        rng (_type_): _description_
        params (_type_): _description_
        num_timesteps (_type_): _description_
        inputs (_type_): _description_
    """
    def _step(carry, rng_and_t):
        state = carry
        rng, t = rng_and_t

        # Shorthand: get parameters and inputs for time index t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(params.dynamics_matrix)
        B = get(params.dynamics_input_weights)
        Q = get(params.dynamics_covariance)
        C = get(params.emission_matrix)
        D = get(params.emission_input_weights)
        R = get(params.emission_covariance)
        u = inputs[t]

        # Sample data and next state
        rng1, rng2 = jr.split(rng, 2)
        emission = MVN(C @ state + D @ u, R).sample(seed=rng1)
        next_state = MVN(A @ state + B @ u, Q).sample(seed=rng2)
        return next_state, (state, emission)

    # Initialize
    rng, this_rng = jr.split(rng, 2)
    init_state = MVN(params.initial_mean, params.initial_covariance).sample(seed=this_rng)

    # Run the sampler
    rngs = jr.split(rng, num_timesteps)
    _, (states, emissions) = lax.scan(
        _step, init_state, (rngs, jnp.arange(num_timesteps)))
    return states, emissions


