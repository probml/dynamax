from functools import partial
from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random
from jax.experimental.host_callback import id_print

from ssm_jax.src.hmm.base import HMMParams


def sampling(key: chex.PRNGKey, hmm_params: HMMParams, *, N: int) -> Tuple[
    chex.Array, chex.Array]:
    """
    Samples from a HMM model given params and a number of required time steps. By default this
    does not sample an observation at the initial time step.
    # TODO: Allow for applying the emission at the initial time step.
    This makes the code a bit convoluted given the need to check a bunch of conditions. Not sure if it's worth the effort.

    Args:
        key:
            A PRNGKey to sample from the HMM.
        hmm_params:
            A HMMParams object defining the HMM.
        N:
            The number of trajectories-observations required.
    Returns:

    """
    return _sampling(key,
                     hmm_params.transition,
                     hmm_params.emission,
                     hmm_params.initial_distribution,
                     hmm_params.time_invariant,
                     hmm_params.T,
                     N)


@partial(jax.jit, static_argnums=(4, 5, 6))
def _sampling(key, transition, emission, pi_0, time_invariant, T, N):
    # Right now we are wasting some uniforms, but this is actually cheaper than propagating keys.
    uniforms = jax.random.uniform(key, (T + 1, 2, N))
    x_0 = _sample_one(pi_0, uniforms[0, 0])

    if time_invariant:
        xs, ys = _time_invariant_loop(x_0, uniforms[1:], transition, emission)
    else:
        xs, ys = _time_varying_loop(x_0, uniforms[1:], transition, emission)
    xs = jnp.insert(xs, 0, x_0, 0)
    return xs, ys


@jax.jit
def _time_invariant_loop(x_0, us, transition, emission):
    def body(x, inp):
        u_x, u_y = inp
        w_x = transition[:, x]
        x = _vmapped_sample_one(w_x, u_x)
        w_y = emission[:, x]
        y = _vmapped_sample_one(w_y, u_y)
        return x, (x, y)

    return jax.lax.scan(body, x_0, us)[1]


@jax.jit
def _time_varying_loop(x_0, us, transition_matrices, emission_matrices):
    def body(x, inp):
        u_x, u_y, transition, emission = inp
        w_x = transition[:, x]

        x = _vmapped_sample_one(w_x, u_x)
        w_y = emission[:, x]
        y = _vmapped_sample_one(w_y, u_y)
        return x, (x, y)

    return jax.lax.scan(body, x_0, (us, transition_matrices, emission_matrices))[1]


def _sample_one(w, u):
    """
    A version of `jax.random.choice` for sampling when we already have access to the uniform.
    Args:
        w:
        u:

    Returns:

    """
    cw = jnp.cumsum(w)
    return jnp.searchsorted(cw, u)


_vmapped_sample_one = jax.vmap(_sample_one, in_axes=[1, 0])
