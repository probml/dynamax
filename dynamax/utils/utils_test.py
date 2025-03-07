"""
Tests of the utility functions.
"""
import jax.numpy as jnp

from dynamax.utils.utils import find_permutation

def test_find_permutation():
    """Test the find_permutation function
    """
    z1 = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    z2 = jnp.array([1, 1, 0, 0, 3, 3, 2, 2, 4, 4])
    true_perm = jnp.array([1, 0, 3, 2, 4])
    perm = find_permutation(z1, z2)
    assert jnp.allclose(jnp.take(perm, z1), z2)
    assert jnp.allclose(true_perm, perm)


def test_find_permutation_unmatched():
    """Test the find_permutation function with K_2 > K_1.
    """
    z1 = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    z2 = jnp.array([5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    true_perm = jnp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    perm = find_permutation(z1, z2)
    assert jnp.allclose(true_perm, perm)


def test_find_permutation_unmatched_v2():
    """Test the find_permutation function with K_2 < K_1.
    """
    z1 = jnp.array([5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    z2 = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    true_perm = jnp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    perm = find_permutation(z1, z2)
    assert jnp.allclose(true_perm, perm)


def test_find_permutation_unmatched_v3():
    """Test the find_permutation function with a more complex assignment.
    """
    z1 = jnp.array([0, 5, 5, 1, 6, 6, 2, 7, 7, 3, 8, 8, 4, 9, 9])
    z2 = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    true_perm = jnp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    perm = find_permutation(z1, z2)
    assert jnp.allclose(true_perm, perm)