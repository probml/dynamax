"""
Utility functions for the library.
"""
import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import jit
from jax import vmap
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
from jaxtyping import Array, Int
from scipy.optimize import linear_sum_assignment
from jax.scipy.linalg import cho_factor, cho_solve

def has_tpu():
    """Check if the current device is a TPU."""
    try:
        return isinstance(jax.devices()[0], jaxlib.xla_extension.TpuDevice)
    except:
        return False


@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    """
    Pad ragged sequences to a fixed length.
    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences
    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence
    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced
    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    """

    def pad(seq, len):
        """Pad a single sequence."""
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)

    dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return dataset


def monotonically_increasing(x, atol=0., rtol=0.):
    """Check if an array is monotonically increasing."""
    thresh = atol + rtol*jnp.abs(x[:-1])
    return jnp.all(jnp.diff(x) >= -thresh)


def pytree_len(pytree):
    """Return the number of leaves in a PyTree."""
    if pytree is None:
        return 0
    else:
        return len(tree_leaves(pytree)[0])


def pytree_sum(pytree, axis=None, keepdims=False, where=None):
    """Sum all the leaves in a PyTree."""
    return tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims, where=where), pytree)


def pytree_slice(pytree, slc):
    """Slice all the leaves in a Pytree."""
    return tree_map(lambda x: x[slc], pytree)


def pytree_stack(pytrees):
    """Stack all the leaves in a list of PyTrees."""
    _, treedef = tree_flatten(pytrees[0])
    leaves = [tree_leaves(tree) for tree in pytrees]
    return tree_unflatten(treedef, [jnp.stack(vals) for vals in zip(*leaves)])

def random_rotation(seed, n, theta=None):
    r"""Helper function to create a rotating linear system.

    Args:
        seed (jax.random.PRNGKey): JAX random seed.
        n (int): Dimension of the rotation matrix.
        theta (float, optional): If specified, this is the angle of the rotation, otherwise
            a random angle sampled from a standard Gaussian scaled by ::math::`\pi / 2`. Defaults to None.
    Returns:
        [type]: [description]
    """

    key1, key2 = jr.split(seed)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * jnp.pi * jr.uniform(key1)

    if n == 1:
        return jr.uniform(key1) * jnp.eye(1)

    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key2, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def ensure_array_has_batch_dim(tree, instance_shapes):
    """Add a batch dimension to a PyTree, if necessary.

    Example: If `tree` is an array of shape (T, D) where `T` is
    the number of time steps and `D` is the emission dimension,
    and if `instance_shapes` is a tuple (D,), then the return
    value is the array with an added batch dimension, with
    shape (1, T, D).

    Example: If `tree` is an array of shape (N,TD) and
    `instance_shapes` is a tuple (D,), then the return
    value is simply `tree`, since it already has a batch
    dimension (of length N).

    Example: If `tree = (A, B)` is a tuple of arrays with
    `A.shape = (100,2)` `B.shape = (100,4)`, and
    `instances_shapes = ((2,), (4,))`, then the return value
    is equivalent to `(jnp.expand_dims(A, 0), jnp.expand_dims(B, 0))`.

    Args:
        tree (_type_): PyTree whose leaves' shapes are either
            (batch, length) + instance_shape or (length,) + instance_shape.
            If the latter, this function adds a batch dimension of 1 to
            each leaf node.

        instance_shape (_type_): matching PyTree where the "leaves" are
            tuples of integers specifying the shape of one "instance" or
            entry in the array.
    """
    def _expand_dim(x, shp):
        """Add a batch dimension to an array, if necessary."""
        ndim = len(shp)
        assert x.ndim > ndim, "array does not match expected shape!"
        assert all([(d1 == d2) for d1, d2 in zip(x.shape[-ndim:], shp)]), \
            "array does not match expected shape!"

        if x.ndim == ndim + 2:
            # x already has a batch dim
            return x
        elif x.ndim == ndim + 1:
            # x has a leading time dimension but no batch dim
            return jnp.expand_dims(x, 0)
        else:
            raise Exception("array has too many dimensions!")

    if tree is None:
        return None
    else:
        return tree_map(_expand_dim, tree, instance_shapes)


def compute_state_overlap(
    z1: Int[Array, " num_timesteps"],
    z2: Int[Array, " num_timesteps"]
):
    """
    Compute a matrix describing the state-wise overlap between two state vectors
    ``z1`` and ``z2``.

    The state vectors should both of shape ``(T,)`` and be integer typed.

    Args:
        z1: The first state vector.
        z2: The second state vector.

    Returns:
        overlap matrix: Matrix of cumulative overlap events.
    """
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K = max(max(z1), max(z2)) + 1

    overlap = jnp.sum(
        (z1[:, None] == jnp.arange(K))[:, :, None]
        & (z2[:, None] == jnp.arange(K))[:, None, :],
        axis=0,
    )
    return overlap


def find_permutation(
    z1: Int[Array, " num_timesteps"],
    z2: Int[Array, " num_timesteps"]
):
    """
    Find the permutation of the state labels in sequence ``z1`` so that they
    best align with the labels in ``z2``.

    Args:
        z1: The first state vector.
        z2: The second state vector.

    Returns:
        permutation such that ``jnp.take(perm, z1)`` best aligns with ``z2``.
        Thus, ``len(perm) = min(z1.max(), z2.max()) + 1``.

    """
    overlap = compute_state_overlap(z1, z2)
    _, perm = linear_sum_assignment(-overlap)
    return perm


def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), b)
    return x

def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))
