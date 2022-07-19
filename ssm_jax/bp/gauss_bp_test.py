from jax import numpy as jnp 

from ssm_jax.bp.gauss_bp import potential_from_conditional_linear_gaussian, info_condition

def test_clg_potential():
    """Test consistency of conditional linear gaussian potential function 
     with joint conditioning function.

    p(y|z) = N(y | Az + u, Lambda^{-1})
    """
    # Parameters
    A = jnp.array([[1., 1., 0., 1.],
                   [0., 1., 2., 3.]])
    z = jnp.ones((4,1))
    u = jnp.ones((2,1))
    Lambda = jnp.ones((2,2)) * 2

    # Form joint potential \phi(y,z)
    K, h = potential_from_conditional_linear_gaussian(A,u,Lambda)
    # Condition on z
    K_cond, h_cond = info_condition(K,h,z)

    # Check that we end up where we started...
    assert jnp.allclose(Lambda, K_cond, rtol=1e-2)
    assert jnp.allclose(Lambda @ (A @ z + u), h_cond, rtol=1e-2)

