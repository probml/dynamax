"""
Hamiltonian Monte Carlo for Bayesian Neural Network
"""

import jax
import distrax
import blackjax
import jax.numpy as jnp
import flax.linen as nn
from chex import dataclass
from typing import Callable
from functools import partial
from jaxtyping import Float, Array, PyTree
from jax.flatten_util import ravel_pytree

@dataclass
class PriorParam:
    scale_obs: float
    scale_weight: float


def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params


def log_joint(
    params: nn.FrozenDict,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    apply_fn: Callable[[PyTree[float], Float[Array, "num_obs dim_obs"]], Float[Array, "num_obs"]],
    priors: PriorParam,
):
    """
    We sample from a BNN posterior assuming
        p(w{i}) = N(0, scale_prior) ∀ i
        P(y | w, X) = N(apply_fn(w, X), scale_obs)

    TODO:
    * Add more general way to compute observation-model log-probability
    """
    scale_obs = priors.scale_obs
    scale_prior = priors.scale_weight
    
    params_flat = get_leaves(params)
    
    # Prior log probability (use initialised vals for mean?)
    logp_prior = distrax.Normal(loc=0.0, scale=scale_prior).log_prob(params_flat).sum()
    
    # Observation log-probability
    mu_obs = apply_fn(params, X).ravel()
    logp_obs = distrax.Normal(loc=mu_obs, scale=scale_obs).log_prob(y).sum()
    
    logprob = logp_prior + logp_obs
    return logprob


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def inference(
    key: jax.random.PRNGKey,
    apply_fn: Callable,
    log_joint: Callable,
    params_init: nn.FrozenDict,
    priors: PriorParam,
    X: Float[Array, "num_obs ..."],
    y: Float[Array, "num_obs"],
    num_warmup: int,
    num_steps: int,
):
    key_warmup, key_train = jax.random.split(key)
    potential = partial(
        log_joint,
        priors=priors, X=X, y=y, apply_fn=apply_fn
    )

    adapt = blackjax.window_adaptation(blackjax.nuts, potential, num_warmup)
    final_state, kernel, _ = adapt.run(key_warmup, params_init)
    states = inference_loop(key_train, kernel, final_state, num_steps)

    return states
