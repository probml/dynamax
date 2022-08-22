from jax import numpy as jnp

from ssm_jax.bp.gauss_bp_utils import potential_from_conditional_linear_gaussian, info_condition

_all_close = lambda x, y: jnp.allclose(x, y, rtol=1e-3, atol=1e-3)


def test_clg_potential():
    """Test consistency of conditional linear gaussian potential function
     with joint conditioning function.

    p(y|z) = N(y | Az + u, Lambda^{-1})
    """
    # Parameters
    A = jnp.array([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
    z = jnp.ones((4, 1))
    offset = jnp.ones((2, 1))
    Lambda = jnp.ones((2, 2)) * 2

    # Form joint potential \phi(y,z)
    (Kzz, Kzy, Kyy), (hz, hy) = potential_from_conditional_linear_gaussian(A, offset, Lambda)
    # Condition on z
    K_cond, h_cond = info_condition(Kyy, Kzy.T, hy, z)

    # Check that conditioning phi(z,y) on z returns the same parameters as 
    #  explicitly calculating linear conditional.
    assert _all_close(Lambda, K_cond)
    assert _all_close(Lambda @ (A @ z + offset), h_cond)
