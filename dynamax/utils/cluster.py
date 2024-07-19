from typing import Tuple

from jax import numpy as jnp
from jax import random as jr

from jaxtyping import Array, Float


def kmeans_sklearn(
    k: int, X: Float[Array, "num_samples state_dim"], key: Array
) -> Tuple[Float[Array, "num_states state_dim"], Float[Array, "num_samples"]]:
    """
    Compute the cluster centers and assignments using the sklearn K-means algorithm.

    Args:
        k (int): The number of clusters.
        X (Array(N, D)): The input data array. N samples of dimension D.
        key (Array): The random seed array.

    Returns:
        Array(k, D), Array(N,): The cluster centers and labels
    """
    from sklearn.cluster import KMeans

    key, subkey = jr.split(key)  # Create a random seed for SKLearn.
    sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
    km = KMeans(k, random_state=int(sklearn_key)).fit(X)
    return jnp.array(km.cluster_centers_), jnp.array(km.labels_)
