from functools import partial
from jax import lax, jit
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Int, Float
from typing import NamedTuple, Tuple


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


class KMeansState(NamedTuple):
    centroids: Float[Array, "num_states state_dim"]
    assignments: Int[Array, "num_samples"]
    prev_centroids: Float[Array, "num_states state_dim"]
    itr: int


@partial(jit, static_argnums=(1, 3))
def kmeans_jax(
    X: Float[Array, "num_samples state_dim"],
    k: int,
    key: Array = jr.PRNGKey(0),
    max_iters: int = 1000,
) -> KMeansState:
    """
    Perform k-means clustering using JAX.

    K-means++ initialization is used to initialize the centroids.

    Args:
        X (Array): The input data array of shape (n_samples, n_features).
        k (int): The number of clusters.
        max_iters (int, optional): The maximum number of iterations. Defaults to 1000.
        key (PRNGKey, optional): The random key for initialization. Defaults to jr.PRNGKey(0).

    Returns:
        KMeansState: A named tuple containing the final centroids array of shape (k, n_features),
        the assignments array of shape (n_samples,) indicating the cluster index for each sample,
        the previous centroids array of shape (k, n_features), and the number of iterations.
    """

    def _update_centroids(X: Array, assignments: Array):
        new_centroids = jnp.array([jnp.mean(X, axis=0, where=(assignments == i)[:, None]) for i in range(k)])
        return new_centroids

    def _update_assignments(X, centroids):
        return jnp.argmin(jnp.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

    def body(carry: KMeansState):
        centroids, assignments, *_ = carry
        new_centroids = _update_centroids(X, assignments)
        new_assignments = _update_assignments(X, new_centroids)
        return KMeansState(new_centroids, new_assignments, centroids, carry.itr + 1)

    def cond(carry: KMeansState):
        return jnp.any(carry.centroids != carry.prev_centroids) & (carry.itr < max_iters)

    def init(key):
        """kmeans++ initialization of centroids

        Iteratively sample new centroids with probability proportional to the squared distance
        from the closest centroid. This initialization method is more stable than random
        initialization and leads to faster convergence.
        Ref: Arthur, D., & Vassilvitskii, S. (2006).
        """
        centroids = jnp.zeros((k, X.shape[1]))
        centroids = centroids.at[0, :].set(jr.choice(key, X))
        for i in range(1, k):
            squared_diffs = jnp.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2)
            min_squared_dists = jnp.min(squared_diffs, axis=1)
            probs = min_squared_dists / jnp.sum(min_squared_dists)
            centroids = centroids.at[i, :].set(jr.choice(key, X, p=probs))
        assignments = _update_assignments(X, centroids)
        # Perform one iteration to update centroids
        updated_centroids = _update_centroids(X, assignments)
        updated_assignments = _update_assignments(X, updated_centroids)
        return KMeansState(updated_centroids, updated_assignments, centroids, 1)

    init_state = init(key)
    state = lax.while_loop(cond, body, init_state)

    return state
