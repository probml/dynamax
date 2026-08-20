"""K-means clustering in JAX, used to initialize HMM emission parameters."""

from functools import partial
from typing import NamedTuple

from jax import jit, lax, vmap
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, Int

from dynamax.types import PRNGKeyT, Scalar


class KMeansState(NamedTuple):
    """Result of a k-means fit.

    Attributes:
        centroids: cluster centers.
        assignments: index of the closest centroid for each sample.
        inertia: sum of squared distances from each sample to its centroid.
        n_iter: number of Lloyd iterations run by the selected restart.
    """

    centroids: Float[Array, "num_clusters num_features"]
    assignments: Int[Array, " num_samples"]
    inertia: Float[Array, ""]
    n_iter: Int[Array, ""]


def _squared_distances(
    X: Float[Array, "num_samples num_features"],
    centroids: Float[Array, "num_clusters num_features"],
) -> Float[Array, "num_samples num_clusters"]:
    """Compute squared euclidean distances from every sample to every centroid.

    Expands ||x - c||^2 to ||x||^2 - 2 x.c + ||c||^2 so that no
    (num_samples, num_clusters, num_features) intermediate is materialized.
    """
    return (
        jnp.sum(X**2, axis=1)[:, None]
        - 2.0 * X @ centroids.T
        + jnp.sum(centroids**2, axis=1)[None, :]
    )


def _assign(
    X: Float[Array, "num_samples num_features"],
    centroids: Float[Array, "num_clusters num_features"],
) -> Int[Array, " num_samples"]:
    """Assign each sample to its closest centroid."""
    return jnp.argmin(_squared_distances(X, centroids), axis=1)


def _update_centroids(
    X: Float[Array, "num_samples num_features"],
    assignments: Int[Array, " num_samples"],
    num_clusters: int,
    previous: Float[Array, "num_clusters num_features"],
) -> Float[Array, "num_clusters num_features"]:
    """Recompute centroids as the mean of their assigned samples.

    A cluster that captured no samples retains its previous centroid. Averaging an
    empty cluster would produce NaN, which propagates into the emission parameters
    and also prevents the fixed-point loop from ever terminating.
    """
    num_features = X.shape[1]
    sums = jnp.zeros((num_clusters, num_features), X.dtype).at[assignments].add(X)
    counts = jnp.zeros((num_clusters,), X.dtype).at[assignments].add(1.0)
    means = sums / jnp.maximum(counts, 1.0)[:, None]
    return jnp.where(counts[:, None] > 0, means, previous)


def _inertia(
    X: Float[Array, "num_samples num_features"],
    centroids: Float[Array, "num_clusters num_features"],
) -> Float[Array, ""]:
    """Sum of squared distances from each sample to its closest centroid."""
    return jnp.sum(jnp.min(_squared_distances(X, centroids), axis=1))


def _kmeans_plusplus(
    key: PRNGKeyT,
    X: Float[Array, "num_samples num_features"],
    num_clusters: int,
) -> Float[Array, "num_clusters num_features"]:
    """Choose initial centroids with k-means++.

    Samples each successive centroid with probability proportional to its squared
    distance from the closest already-chosen centroid. This spreads the initial
    centroids out, which converges faster and more reliably than a uniform draw.
    Ref: Arthur, D., & Vassilvitskii, S. (2006). "k-means++: the advantages of
    careful seeding."
    """
    num_samples, num_features = X.shape
    key, subkey = jr.split(key)
    centroids = jnp.zeros((num_clusters, num_features), X.dtype).at[0].set(jr.choice(subkey, X))

    def step(carry, _):
        """Sample one additional centroid proportional to squared distance."""
        centroids, i, key = carry
        key, subkey = jr.split(key)
        # Mask the not-yet-chosen centroid slots so their zeros do not skew distances.
        distances = jnp.where(
            (jnp.arange(num_clusters) < i)[None, :],
            _squared_distances(X, centroids),
            jnp.inf,
        )
        min_distances = jnp.min(distances, axis=1)
        total = jnp.sum(min_distances)
        # If every sample already sits on a centroid, fall back to a uniform draw.
        probs = jnp.where(
            total > 0,
            min_distances / jnp.where(total > 0, total, 1.0),
            jnp.ones_like(min_distances) / num_samples,
        )
        centroid = jr.choice(subkey, X, p=probs)
        return (centroids.at[i].set(centroid), i + 1, key), None

    (centroids, _, _), _ = lax.scan(step, (centroids, 1, key), None, length=num_clusters - 1)
    return centroids


@partial(jit, static_argnames=("k", "max_iters", "n_init"))
def kmeans(
    X: Float[Array, "num_samples num_features"],
    k: int,
    key: PRNGKeyT,
    max_iters: int = 100,
    tol: Scalar = 1e-6,
    n_init: int = 10,
) -> KMeansState:
    """Cluster `X` into `k` groups with Lloyd's algorithm and k-means++ seeding.

    Runs `n_init` independent restarts and returns the one with the lowest inertia,
    because a single restart can settle in a poor local optimum. Restarts are
    vectorized with `vmap`, so they cost little more than one run on an accelerator.

    Args:
        X: samples to cluster.
        k: number of clusters. Static: changing it triggers recompilation.
        key: random seed for k-means++ initialization.
        max_iters: cap on Lloyd iterations per restart. Static.
        tol: stop once an iteration improves inertia by no more than this.
        n_init: number of independent restarts. Static.

    Returns:
        The best `KMeansState` across restarts.
    """

    def single_run(key: PRNGKeyT) -> KMeansState:
        """Run one restart of Lloyd's algorithm from a k-means++ seeding."""

        def cond(carry):
            """Continue while inertia is still improving by more than tol."""
            _, previous_inertia, inertia, i = carry
            return (i < max_iters) & (previous_inertia - inertia > tol)

        def body(carry):
            """Run one Lloyd iteration: reassign samples, then recompute centroids."""
            centroids, _, inertia, i = carry
            assignments = _assign(X, centroids)
            new_centroids = _update_centroids(X, assignments, k, centroids)
            return new_centroids, inertia, _inertia(X, new_centroids), i + 1

        initial = _kmeans_plusplus(key, X, k)
        centroids = _update_centroids(X, _assign(X, initial), k, initial)
        carry = (centroids, jnp.inf, _inertia(X, centroids), 1)
        centroids, _, inertia, n_iter = lax.while_loop(cond, body, carry)
        return KMeansState(centroids, _assign(X, centroids), inertia, n_iter)

    restarts = vmap(single_run)(jr.split(key, n_init))
    best = jnp.argmin(restarts.inertia)
    return KMeansState(
        restarts.centroids[best],
        restarts.assignments[best],
        restarts.inertia[best],
        restarts.n_iter[best],
    )
