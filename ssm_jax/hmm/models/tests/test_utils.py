import jax.numpy as jnp


def normalize(a, axis=None):
    """
    Normalize the input array so that it sums to 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = jnp.sum(a, axis=axis, keepdims=True)
    a_sum = jnp.where(a_sum == 0, 1, a_sum)
    return a / a_sum


def normalized(X, axis=None):
    return normalize(X, axis=axis)
