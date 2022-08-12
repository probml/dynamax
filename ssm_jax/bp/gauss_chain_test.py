from jax import numpy as jnp
from jax import random as jr

from ssm_jax.bp.gauss_chain import gauss_chain_potentials_from_lgssm, gauss_chain_bp
from ssm_jax.lgssm.info_inference import lgssm_info_smoother
from ssm_jax.lgssm.info_inference_test import build_lgssm_moment_and_info_form

def test_gauss_chain_bp():
    """Test that Gaussian chain belief propagation gets the same results as 
     information form RTS smoother."""

    lgssm, lgssm_info = build_lgssm_moment_and_info_form()

    key = jr.PRNGKey(111)
    num_timesteps = 15
    input_size = lgssm.dynamics_input_weights.shape[1]
    inputs = jnp.zeros((num_timesteps, input_size))
    z, y = lgssm.sample(key, num_timesteps, inputs)

    lgssm_info_posterior = lgssm_info_smoother(lgssm_info, y, inputs)

    prior_pot, chain_pots = gauss_chain_potentials_from_lgssm(lgssm_info, inputs)

    bels = gauss_chain_bp(chain_pots, prior_pot, y)
    Ks, hs = bels

    assert jnp.allclose(lgssm_info_posterior.smoothed_precisions,Ks,
                       rtol=1e-3)
    assert jnp.allclose(lgssm_info_posterior.smoothed_etas,hs,
                       rtol=1e-3)

