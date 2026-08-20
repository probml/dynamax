"""Tests for the k-means clustering utilities."""

import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap

from dynamax.utils.cluster import kmeans


def test_kmeans_recovers_toy_optimum():
    """Recovers the known optimal 2-means solution on a 4-point toy example."""
    x = jnp.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [1.0, 1.0]])
    state = kmeans(x, 2, jr.PRNGKey(101))
    # Compare permutation-invariantly by sorting centroids on their y coordinate.
    centroids = state.centroids[jnp.argsort(state.centroids[:, 1])]
    assert jnp.allclose(centroids, jnp.array([[0.25, 0.0], [0.75, 1.0]]), atol=1e-5)
    assert jnp.allclose(state.inertia, 0.25, atol=1e-5)


def test_kmeans_recovers_separated_clusters():
    """Assignments perfectly partition well-separated blobs."""
    key = jr.PRNGKey(0)
    means = jnp.array([[-6.0, -6.0], [0.0, 0.0], [6.0, 6.0]])
    x = jnp.concatenate([m + 0.3 * jr.normal(k, (40, 2)) for m, k in zip(means, jr.split(key, 3))])
    state = kmeans(x, 3, jr.PRNGKey(1))
    true_labels = jnp.repeat(jnp.arange(3), 40)
    # Each true blob must map onto exactly one predicted label.
    for blob in range(3):
        assert jnp.unique(state.assignments[true_labels == blob]).size == 1
    assert jnp.unique(state.assignments).size == 3


def test_kmeans_empty_clusters_do_not_produce_nans():
    """k exceeding the number of distinct points must not yield NaN centroids."""
    x = jnp.ones((4, 2))
    state = kmeans(x, 3, jr.PRNGKey(0))
    assert not jnp.any(jnp.isnan(state.centroids))
    assert jnp.allclose(state.inertia, 0.0)
    assert int(state.n_iter) < 10  # terminates promptly instead of spinning to max_iters


def test_kmeans_k_greater_than_distinct_points():
    """Degenerate data with fewer distinct points than clusters stays finite."""
    x = jnp.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    state = kmeans(x, 3, jr.PRNGKey(0))
    assert jnp.all(jnp.isfinite(state.centroids))
    assert jnp.all(jnp.isfinite(state.inertia))


def test_kmeans_one_dimensional_data():
    """Works on (N, 1) data, the shape GammaHMM passes in."""
    x = jnp.concatenate([jr.normal(jr.PRNGKey(1), (50, 1)) + 5, jr.normal(jr.PRNGKey(2), (50, 1)) - 5])
    state = kmeans(x, 2, jr.PRNGKey(0))
    centroids = jnp.sort(state.centroids.ravel())
    assert centroids[0] < 0 < centroids[1]
    assert jnp.unique(state.assignments).size == 2


def test_kmeans_inertia_matches_assignments():
    """Reported inertia equals the sum of squared distances to assigned centroids."""
    x = jr.normal(jr.PRNGKey(4), (100, 3))
    state = kmeans(x, 4, jr.PRNGKey(5))
    recomputed = jnp.sum((x - state.centroids[state.assignments]) ** 2)
    assert jnp.allclose(state.inertia, recomputed, rtol=1e-5)


def test_kmeans_is_deterministic_given_key():
    """The same key produces identical results."""
    x = jr.normal(jr.PRNGKey(6), (80, 2))
    a = kmeans(x, 3, jr.PRNGKey(7))
    b = kmeans(x, 3, jr.PRNGKey(7))
    assert jnp.allclose(a.centroids, b.centroids)
    assert jnp.all(a.assignments == b.assignments)


def test_kmeans_restarts_beat_single_init():
    """n_init restarts find an optimum at least as good as a single restart."""
    key = jr.PRNGKey(0)
    means = jnp.array([[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0], [8.0, -4.0]])
    x = jnp.concatenate([m + 0.6 * jr.normal(k, (60, 2)) for m, k in zip(means, jr.split(key, 4))])
    # Seed 3 lands in a bad local optimum with a single initialization.
    single = kmeans(x, 4, jr.PRNGKey(3), n_init=1)
    many = kmeans(x, 4, jr.PRNGKey(3), n_init=10)
    assert many.inertia <= single.inertia
    assert many.inertia < 200.0  # the good optimum; the bad one is ~1045


def test_kmeans_is_jittable_and_vmappable():
    """Works under jit and vmap over a batch of datasets."""
    x = jr.normal(jr.PRNGKey(8), (4, 60, 2)) * 2
    keys = jr.split(jr.PRNGKey(9), 4)
    batched = vmap(lambda d, k: kmeans(d, 3, k))(x, keys)
    assert batched.centroids.shape == (4, 3, 2)
    assert batched.assignments.shape == (4, 60)
    assert jnp.all(jnp.isfinite(batched.inertia))

    single = jit(lambda d, k: kmeans(d, 3, k))(x[0], keys[0])
    assert jnp.allclose(single.centroids, batched.centroids[0])
