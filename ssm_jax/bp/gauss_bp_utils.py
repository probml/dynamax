from functools import partial
from jax import jit
from jax import numpy as jnp
from jax.tree_util import tree_map


def info_marginalize(Kxx, Kxy, Kyy, hx, hy):
    """Calculate the parameters of marginalized MVN.

    For x, y joint distributed as
        p(x, y) = Nc(x,y| h, K),
    the marginal distribution of x is given by:
        p(y) = \int p(x, y) dx = Nc(y | hy_marg, Ky_marg)
    where,
        hy_marg = hy - Kyx Kxx^{-1} hx
        Ky_marg = Kyy - Kyx Kxx^{-1} Kxy

    Args:
        K_blocks: blocks of the joint precision matrix, (Kxx, Kxy, Kyy),
                    Kxx (dim_x, dim_x),
                    Kxy (dim_x, dim_y),
                    Kyy (dim_y, dim_y).
        hs (dim_x + dim_y, 1): joint precision weighted mean, (hx, hy):
                    h1 (dim_x, 1),
                    h2 (dim_y, 1).
    Returns:
        Ky_marg (dim_y, dim_y): marginal precision matrix.
        hy_marg (dim_y,1): marginal precision weighted mean.
    """
    G = jnp.linalg.solve(Kxx, Kxy)
    Ky_marg = Kyy - Kxy.T @ G
    hy_marg = hy - G.T @ hx
    return Ky_marg, hy_marg


def info_condition(Kxx, Kxy, hx, y):
    """Calculate the parameters of MVN after conditioning.

    For x,y with joint mvn
        p(x,y) = Nc(x,y | h, K),
    where h, K can be partitioned into,
        h = [hx, hy]
        K = [[Kxx, Kxy],
            [[Kyx, Kyy]]
    the distribution of x condition on a particular value of y is given by,
        p(x|y) = Nc(x | hx_cond, Kx_cond),
    where
        hx_cond = hx - Kxy y
        Kx_cond = Kxx
    """
    return Kxx, hx - Kxy @ y


def potential_from_conditional_linear_gaussian(A, offset, Lambda):
    """Express a conditional linear Gaussian as a potential in canonical form.

    p(y|z) = N(y | Az + offset, Lambda^{-1})
           \prop exp( -0.5(y z)^T K (y z) + (y z)^T h )
    where,
        K = (Lambda; -Lambda A,  -A.T Lambda; A.T Lambda A)
        h = (Lambda offset, -A.T Lambda offset)

    Args:
        A (dim_y, dim_z)
        offset (dim_y,1)
        Lambda (dim_y, dim_y)
    Returns:
        K (dim_z + dim_y, dim_z + dim_y)
        h (dim_z + dim_y,1)
    """
    Kzy = -A.T @ Lambda
    Kzz = -Kzy @ A
    Kyy = Lambda
    hy = Lambda @ offset
    hz = -A.T @ hy
    return (Kzz, Kzy, Kyy), (hz, hy)


def info_multiply(params1, params2):
    """Calculate parameters resulting from multiplying Gaussians potentials.

    As all the resultant parameters are the sum of the parameters of the two
     potentials being multiplied, then `params1` and `params2` can be any 
     PyTree of potential parameters as long as the corresponding parameters 
     of the two input potentials occupy the same leaves of the PyTree.

    For example, 
        phi(K1,h2) * phi(K2, h2) = phi(K1 + K2, h1 + h2)

    Args:
        params1: PyTree of potential parameters.
        params2: PyTree of potential parameters with the same tree structure
                  as `params1`.

    Returns:
        params_out: PyTree of resultant potential parameters.
    """
    return tree_map(lambda a, b: a + b, params1, params2)


def info_divide(params1, params2):
    """Calculate parameters resulting from dividing Gaussian potentials.

    As all the resultant parameters are the difference between the parameters 
     of the two potentials being divided, then `params1` and `params2` can be
     any PyTree of potential parameters as long as the corresponding parameters 
     of the two input potentials occupy the same leaves of the PyTree.

    For example, 
        phi(K1,h2) / phi(K2, h2) = phi(K1 - K2, h1 - h2)

    Args:
        params1: PyTree of potential parameters.
        params2: PyTree of potential parameters with the same tree structure
                  as `params1`.

    Returns:
        params_out: PyTree of resultant potential parameters.
    """
    return tree_map(lambda a, b: a - b, params1, params2)


@partial(jit, static_argnums=2)
def pair_cpot_condition(cpot, obs, obs_var):
    """Convenience function for conditioning Gaussian potentials involving two
     variables.

    Args:
        cpot: canonical parameters of the potential, stored as nested tuples
                of the form,
                  ((K11, K12, K22), (h1, h2)).
        obs: observation.
        obs_var (int): the label of the variable being condition on.

    Returns:
        cond_pot: canonical parameters of the conditioned potential,
                    (K_cond, h_cond).
    """
    (K11, K12, K22), (h1, h2) = cpot
    if obs_var == 1:
        return info_condition(K22, K12.T, h2, obs)
    elif obs_var == 2:
        return info_condition(K11, K12, h1, obs)
    else:
        raise ValueError("obs_var must take a value of either 1 or 2.")


@partial(jit, static_argnums=1)
def pair_cpot_marginalize(cpot, marginalize_onto):
    """Convenience function for marginalizing Gaussian potentials involving two
     variables.

    Args:
        cpot: canonical parameters of the potential, stored as nested tuples
                of the form,
                  ((K11, K12, K22), (h1, h2)).
        marginalize_onto (int): the label of the output marginal variable.

    Returns:
        marg_pot: canonical parameters of the marginal potential,
                    (K_marg, h_marg).
    """
    (K11, K12, K22), (h1, h2) = cpot
    if marginalize_onto == 1:
        return info_marginalize(K22, K12.T, K11, h2, h1)
    elif marginalize_onto == 2:
        return info_marginalize(K11, K12, K22, h1, h2)
    else:
        raise ValueError("marg_var must take a value of either 1 or 2.")


@partial(jit, static_argnums=2)
def pair_cpot_absorb_message(cpot, message, message_var):
    """Convenience function for absorbing a message into a Gaussain potential
     involving two variables.

    Args:
        cpot: canonical parameters of the potential, stored as nested tuples
                of the form,
                  ((K11, K12, K22), (h1, h1)).
        message: the message potential which takes the form,
                  (K_message, h_message)
        message_var (int): the label of the output marginal variable.

    Returns:
        cpot_plus_message: canonical parameters of the joint potential after
                            the message has been incorporated,
                             ((K11, K12, K22), (h1, h2))
    """
    K_message, h_message = message
    if message_var == 1:
        padded_message = ((K_message, 0, 0), (h_message, 0))
    elif message_var == 2:
        padded_message = ((0, 0, K_message), (0, h_message))
    else:
        raise ValueError("message_var must take a value of either 1 or 2.")

    return info_multiply(cpot, padded_message)
