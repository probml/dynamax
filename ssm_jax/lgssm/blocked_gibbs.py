import jax.numpy as jnp
import jax.random as jr

from inference import lgssm_posterior_sample


def _lgssm_parameters_sample(rng, states):
    rng = None
    return parameters


def lgssm_blocked_gibbs(rng, emissions, prior, inputs=None):
    ll, states = lgssm_posterior_sample(rng, emissions, inputs)
    params = _lgssm_parameters_sample(rng, states)
    log_evidence = ll + 0
    return params, states, log_evidence