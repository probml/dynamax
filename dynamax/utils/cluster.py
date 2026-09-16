"""K-means clustering in JAX, used to initialize HMM emission parameters."""

import math
from functools import partial
from typing import NamedTuple, Optional

from jax import jit, lax
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
    Centers both operands on the sample mean first: the uncentered expansion
    is not offset-stable in float32 and returns negative "squared" distances
    once the data sits a few orders of magnitude from the origin, which
    silently corrupts both clustering and n_init restart selection.
    """
    offset = jnp.mean(X, axis=0)
    centered_X = X - offset
    centered_centroids = centroids - offset
    return jnp.maximum(
        jnp.sum(centered_X**2, axis=1)[:, None]
        - 2.0 * centered_X @ centered_centroids.T
        + jnp.sum(centered_centroids**2, axis=1)[None, :],
        0.0,
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
    counts = jnp.zeros((num_clusters,), X.dtype).at[assignments].add(jnp.ones((), X.dtype))
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
    n_local_trials: int,
) -> Float[Array, "num_clusters num_features"]:
    """Choose initial centroids with greedy k-means++.

    Samples each successive centroid with probability proportional to its squared
    distance from the closest already-chosen centroid. This spreads the initial
    centroids out, which converges faster and more reliably than a uniform draw.
    Ref: Arthur, D., & Vassilvitskii, S. (2006). "k-means++: the advantages of
    careful seeding."

    Each step draws `n_local_trials` candidates from that distribution rather than
    one, and keeps whichever candidate leaves the lowest total inertia. Drawing a
    single candidate leaves the seeding at the mercy of one unlucky draw: on
    well-separated blobs it lands in a bad local optimum on roughly a third of
    seeds, and recovering from that needs about ten restarts. Greedy selection
    removes those failures at a cost of one extra `(num_samples, n_local_trials)`
    distance block per step. Note that trials are not a substitute for restarts --
    every candidate is scored against the same already-chosen prefix, so no number
    of trials can undo a bad early commitment.
    """
    num_samples, num_features = X.shape
    key, subkey = jr.split(key)
    first = jr.choice(subkey, X)
    centroids = jnp.zeros((num_clusters, num_features), X.dtype).at[0].set(first)
    # Distance from every sample to its closest chosen centroid, carried forward
    # rather than recomputed. Recomputing costs a (num_samples, num_clusters) block
    # per step; carrying it costs a (num_samples, n_local_trials) block, which is
    # smaller as soon as n_local_trials < num_clusters and does not grow with k.
    closest = _squared_distances(X, first[None, :])[:, 0]

    def step(carry, i):
        """Draw several candidate centroids and keep the one minimizing inertia."""
        centroids, closest, key = carry
        key, subkey = jr.split(key)
        total = jnp.sum(closest)
        # If every sample already sits on a centroid, fall back to a uniform draw.
        probs = jnp.where(
            total > 0,
            closest / jnp.where(total > 0, total, 1.0),
            jnp.full((num_samples,), 1.0 / num_samples, closest.dtype),
        )
        candidate_ids = jr.choice(subkey, num_samples, shape=(n_local_trials,), p=probs)
        candidates = X[candidate_ids]
        # Column j holds what `closest` would become if candidate j were chosen.
        distances = jnp.minimum(closest[:, None], _squared_distances(X, candidates))
        best = jnp.argmin(jnp.sum(distances, axis=0))
        return (centroids.at[i].set(candidates[best]), distances[:, best], key), None

    (centroids, _, _), _ = lax.scan(step, (centroids, closest, key), jnp.arange(1, num_clusters))
    return centroids


@partial(jit, static_argnames=("k", "max_iters", "n_init", "n_local_trials"))
def kmeans(
    X: Float[Array, "num_samples num_features"],
    k: int,
    key: PRNGKeyT,
    max_iters: int = 100,
    tol: Scalar = 1e-6,
    n_init: int = 3,
    n_local_trials: Optional[int] = None,
) -> KMeansState:
    """Cluster `X` into `k` groups with Lloyd's algorithm and k-means++ seeding.

    Runs `n_init` independent restarts and returns the one with the lowest inertia,
    because a single restart can settle in a poor local optimum. Restarts run
    sequentially via `lax.map` rather than as one vectorized batch: batching
    materializes a `(n_init, num_samples, k)` distance temporary, whereas
    `lax.map` stacks only each restart's returned state, which is dominated by
    the `(num_samples,)` assignments whenever `num_samples` exceeds
    `k * num_features`. Peak memory therefore grows roughly `k` times more
    slowly in `n_init`. On CPU this is also faster, since each restart exits at
    its own convergence instead of the whole batch running until the slowest
    one converges. Greedy seeding widens that gap rather than closing it: it
    makes most restarts converge in a handful of iterations while leaving the
    occasional unlucky one slow, so per-restart iteration counts spread out
    (measured at N=100k, k=10: [3, 3, 4, 4, 4, 5, 5, 48, 48, 91]) and a batched
    `while_loop`, which must run every lane until the slowest lane stops,
    wastes proportionally more work.

    `n_init` defaults to 3 rather than 1 because greedy seeding does not make
    restarts redundant. All of a step's candidates are scored against the same
    already-chosen centroids, so no number of trials can undo a bad early
    commitment; only a fresh restart resamples it. Measured over 20 seeds, one
    restart still reaches a bad optimum on roughly half of small problems
    (N=1000, D=2, k=5), while three restarts match or beat the quality of the
    ten restarts this previously defaulted to, on every workload tested.

    Args:
        X: samples to cluster.
        k: number of clusters. Static: changing it triggers recompilation.
        key: random seed for k-means++ initialization.
        max_iters: cap on Lloyd iterations per restart. Static.
        tol: stop once an iteration improves inertia by no more than this.
        n_init: number of independent restarts. Static.
        n_local_trials: candidate centroids evaluated per k-means++ step. Defaults to
            `2 + int(log(k))`. Higher values improve seeding with diminishing returns
            and are not a substitute for `n_init`. Static.

    Returns:
        The best `KMeansState` across restarts.
    """

    trials = 2 + int(math.log(k)) if n_local_trials is None else n_local_trials

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

        initial = _kmeans_plusplus(key, X, k, trials)
        centroids = _update_centroids(X, _assign(X, initial), k, initial)
        carry = (centroids, jnp.inf, _inertia(X, centroids), 1)
        centroids, _, inertia, n_iter = lax.while_loop(cond, body, carry)
        return KMeansState(centroids, _assign(X, centroids), inertia, n_iter)

    restarts = lax.map(single_run, jr.split(key, n_init))
    best = jnp.argmin(restarts.inertia)
    return KMeansState(
        restarts.centroids[best],
        restarts.assignments[best],
        restarts.inertia[best],
        restarts.n_iter[best],
    )
