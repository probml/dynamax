from jax import numpy as jnp
from jax import random as jr

from ssm_jax.bp.gauss_chain import gauss_chain_potentials_from_lgssm, gauss_chain_bp
from ssm_jax.linear_gaussian_ssm.inference import lgssm_sample
from ssm_jax.linear_gaussian_ssm.info_inference import lgssm_info_smoother
from ssm_jax.linear_gaussian_ssm.info_inference_test import build_lgssm_moment_and_info_form

_all_close = lambda x,y: jnp.allclose(x,y,rtol=1e-3, atol=1e-3)

def test_gauss_chain_bp():
    """Test that Gaussian chain belief propagation gets the same results as
     information form RTS smoother."""

    lgssm, lgssm_info = build_lgssm_moment_and_info_form()

    key = jr.PRNGKey(111)
    num_timesteps = 15
    input_size = lgssm.dynamics_input_weights.shape[1]
    inputs = jnp.zeros((num_timesteps, input_size))
    x, y = lgssm_sample(key, lgssm, num_timesteps, inputs=inputs)

    lgssm_info_posterior = lgssm_info_smoother(lgssm_info, y, inputs)

    chain_pots = gauss_chain_potentials_from_lgssm(lgssm_info, inputs)

    smoothed_bels = gauss_chain_bp(chain_pots, y)
    Ks, hs = smoothed_bels

    assert _all_close(lgssm_info_posterior.smoothed_precisions,Ks)
    assert _all_close(lgssm_info_posterior.smoothed_etas,hs)


