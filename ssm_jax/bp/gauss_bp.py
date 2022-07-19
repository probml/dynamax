from jax import numpy as jnp
from jax import tree_map

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
    vblocks = jnp.vsplit(A,split_array)
    # [leaf for tree in forest for leaf in tree]
    blocks = [block for vblock in vblocks for block in jnp.hsplit(vblock,split_array)]
    # Can also do:
    #   blocks = tree_map(lambda arr: jnp.hsplit(arr,split_array),vblocks)
    # followed by a tree_flatten
    return blocks


def block_join(A11, A12, A21, A22):
    return jnp.block([[A11, A12],[A21,A22]])

def block_rev(A,idx):
    blocks = block_split(A,idx)
    return block_join(*blocks[::-1])

def vec_swap(a,idx):
    return jnp.concatenate((a[idx:],a[:idx]))

def info_marginalise(K_blocks, hs):
    """Calculate the parameters of marginalised MVN.
    
    For x1, x2 joint distributed as
        p(x1, x2) = Nc(x1,x2| h, K),
    the marginal distribution of x1 is given by:
        p(x2) = \int p(x1, x2) dx1 = Nc(x2 | h2_marg, K2_marg)
    where,
        h2_marg = h2 - K21 K11^{-1} h1
        K2_marg = K22 - K21 K11^{-1} K12

    Args:
        K_blocks: blocks of the joint precision matrix, (K1, K12, K22),
                    K1 (dim1,dim1),
                    K12 (dim1, dim2),
                    K22 (dim2, dim2).
        hs (D,1): joint precision weighted mean, (h1, h2):
                    h1 (dim1, 1),
                    h2 (dim2, 1).
    Returns:
        K2_marg (dim2, dim2): marginal precision matrix.
        h2_marg (dim2,1): marginal precision weighted mean.
    """
    K11, K12, K22 = K_blocks
    h1, h2 = hs 
    G = jnp.linalg.solve(K11,K12)
    K2_marg = K22 - K12.T @ G
    h2_marg = h2 - G.T @ h1
    return K2_marg, h2_marg

def info_condition(K11, K12, h1, y):
    # TODO: Decide on (x1, x2) vs (x,y)
    """Calculate the parameters of MVN after conditioning.

    For x,y with joint mvn
        p(x1,x2) = Nc(x1,x2 | h, K),
    where h, K can be partitioned into,
        h = [h1, h2]
        K = [[K11, K12],
            [[K21, K22]]
    the distribution of x condition on a particular value of y is given by,
        p(x1|x2) = Nc(x1 | h1_cond, K1_cond),
    where
        h1_cond = h1 - K12 x2
        K1_cond = K11
    """
    return K11, h1 - K12 @ y

def potential_from_conditional_linear_gaussian(A,u,Lambda):
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
    Kyy = Lambda 
    Kyz  = -Lambda @ A
    Kzz = -A.T @ Kyz
    hy = Lambda @ u 
    hz = -A.T @ hy
    # TODO: Might be more natural to frame this the other way round, return. (x2 | x1)
    #        or at least just return Kzy because we tend to want to condition on y.
    return (Kyy, Kyz, Kzz), (hy,hz)


def info_multiply(params1, params2):
    """Calculate parameters resulting from multiplying gaussians."""
    return tree_map(lambda a,b: a + b, params1, params2)


def info_divide(params1, params2):
    """Calculate parameters resulting from dividing gaussians."""
    return tree_map(lambda a,b: a - b, params1, params2)

