from jax import numpy as jnp
from jax import random as jr
from jax import vmap

from dynamax.utils.cluster import kmeans_jax


def test_kmeans_jax_toy():
    """Checks that kmeans works against toy example.

    Ref: scikit-learn tests
    """

    key = jr.PRNGKey(101)
    x = jnp.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])

    centroids, assignments, *_ = kmeans_jax(x, 2, key)

    # There are two possible solutions for the centroids and assignments
    try:
        expected_labels = jnp.array([0, 0, 1, 1])
        expected_centers = jnp.array([[0.25, 0], [0.75, 1]])
        assert jnp.all(assignments == expected_labels)
        assert jnp.allclose(centroids, expected_centers)
    except AssertionError:
        expected_labels = jnp.array([1, 1, 0, 0])
        expected_centers = jnp.array([[0.75, 1.0], [0.25, 0.0]])
        assert jnp.all(assignments == expected_labels)
        assert jnp.allclose(centroids, expected_centers)


def test_kmeans_jax_vmap():
    """Test that kmeans_jax works with vmap."""

    def _gen_data(key):
        """Generate 3 clusters of 10 samples each."""
        subkeys = jr.split(key, 3)
        means = jnp.array([-2., 0., 2.])
        _2D_normal = lambda key, mean: jr.normal(key, (10, 2))*0.2 + mean
        return vmap(_2D_normal)(subkeys, means).reshape(-1, 2)

    key = jr.PRNGKey(5)
    key, *data_subkeys = jr.split(key,3)
    # Generate 2 samples of the 3-cluster data
    x = vmap(_gen_data)(jnp.array(data_subkeys))

    alg_subkeys = jr.split(key, 2)
    _, assignments, *_ = vmap(kmeans_jax, (0, None, 0))(x, 3, alg_subkeys)
    # Check that the assignments are the same for both samples (clusters are very distinct)
    assert jnp.all(assignments[0] == assignments[1])
