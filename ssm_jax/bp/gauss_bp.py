from functools import partial
from jax import jit
from jax import numpy as jnp
from jax.tree_util import tree_map


def block_split(A, idx):
    """Split square matrix A into four blocks at `idx`

    For example splitting
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7 ,8]]
    at idx = 2 produces:
        ([[0,1],[3,4]],
         [[2],[5]],
         [[6,7]],
         [[8]]).

    Args:
        A (D, D): array.
        idx: int.
    Returns:
        A11 (dim1, dim1)
        A12 (dim2, dim1)
        A21 (dim1, dim2)
        A22 (dim2, dim2)
       where `dim1 + dim2 == D` and `dim2 = D - idx`.
    """
    split_array = jnp.array([idx])
    vblocks = jnp.vsplit(A, split_array)
    # [leaf for tree in forest for leaf in tree]
    blocks = [block for vblock in vblocks for block in jnp.hsplit(vblock, split_array)]
    # Can also do:
    #   blocks = tree_map(lambda arr: jnp.hsplit(arr,split_array),vblocks)
    # followed by a tree_flatten
    return blocks


def block_join(A11, A12, A21, A22):
    return jnp.block([[A11, A12], [A21, A22]])


def block_rev(A, idx):
    blocks = block_split(A, idx)
    return block_join(*blocks[::-1])


def vec_swap(a, idx):
    return jnp.concatenate((a[idx:], a[:idx]))


def info_marginalise(K_blocks, hs):
    """Calculate the parameters of marginalised MVN.

    For x, y joint distributed as
        p(x, y) = Nc(x,y| h, K),
    the marginal distribution of x is given by:
        p(y) = \int p(x, y) dx = Nc(y | hy_marg, Ky_marg)
    where,
        hy_marg = h2 - Kyx Kxx^{-1} h1
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
    Kxx, Kxy, Kyy = K_blocks
    hx, hy = hs
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


def potential_from_conditional_linear_gaussian(A, u, Lambda):
    """Express a conditional linear gaussian as a potential in canonical form.

    p(y|z) = N(y | Az + u, Lambda^{-1})
           \prop exp( -0.5(y z)^T K (y z) + (y z)^T h )
    where,
        K = (Lambda; -Lambda A,  -A.T Lambda; A.T Lambda A)
        h = (Lambda u, -A.T Lambda u)

    Args:
        A (dim1, dim2)
        u (dim1,1)
        Lambda (dim1, dim1)
    Returns:
        K (dim1 + dim2, dim1 + dim2)
        h (dim1 + dim2,1)
    """
    Kzy = -A.T @ Lambda
    Kzz = -Kzy @ A
    Kyy = Lambda
    hy = Lambda @ u
    hz = -A.T @ hy
    return (Kzz, Kzy, Kyy), (hz, hy)


def info_multiply(params1, params2):
    """Calculate parameters resulting from multiplying gaussians."""
    return tree_map(lambda a, b: a + b, params1, params2)


def info_divide(params1, params2):
    """Calculate parameters resulting from dividing gaussians."""
    return tree_map(lambda a, b: a - b, params1, params2)


@partial(jit, static_argnums=2)
def pair_cpot_condition(cpot, obs, obs_var):
    """Convenience function for conditioning gaussian potentials involving two
     variables.

    Args:
        cpot: canonical parameters of the potential, stored as nested tuples
                of the form,
                  ((K11, K12, K22), (h1, h1)).
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
def pair_cpot_marginalise(cpot, marg_var):
    """Convenience function for marginalising gaussian potentials involving two
     variables.

    Args:
        cpot: canonical parameters of the potential, stored as nested tuples
                of the form,
                  ((K11, K12, K22), (h1, h1)).
        marg_var (int): the label of the output marginal variable.

    Returns:
        marg_pot: canonical parameters of the marginal potential,
                    (K_marg, h_marg).
    """
    if marg_var == 1:
        (K11, K12, K22), (h1, h2) = cpot
        return info_marginalise((K22, K12.T, K11), (h2, h1))
    elif marg_var == 2:
        return info_marginalise(*cpot)
    else:
        raise ValueError("marg_var must take a value of either 1 or 2.")


@partial(jit, static_argnums=2)
def pair_cpot_absorb_message(cpot, message, message_var):
    """Convenience function for absorbing a message into a gaussain potential
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
