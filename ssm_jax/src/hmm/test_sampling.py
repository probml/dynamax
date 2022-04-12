import chex
import jax.numpy as jnp
import jax.random
import numpy as np

from ssm_jax.src.hmm.base import HMMParams
from ssm_jax.src.hmm.sampling import sampling


def test_sampling_time_invariant():
    np.random.seed(0)
    key = jax.random.PRNGKey(42)
    T = 25
    N = 100_000
    d_x = 5
    d_y = 3

    pi_0 = np.random.rand(d_x)
    pi_0 = pi_0 / np.sum(pi_0, 0)

    transition = np.random.rand(d_x, d_x)
    transition = transition / np.sum(transition, axis=0, keepdims=True)

    emission = np.random.rand(d_y, d_x)
    emission = emission / np.sum(emission, axis=0, keepdims=True)

    hmm_params = HMMParams(pi_0, transition, emission, True, T)
    xs, ys = sampling(key, hmm_params=hmm_params, N=N)

    assert xs.shape == (T + 1, N)
    assert ys.shape == (T, N)

    actual_x_probas = jax.vmap(lambda z: jnp.bincount(z, length=d_x))(xs) / N
    actual_y_probas = jax.vmap(lambda z: jnp.bincount(z, length=d_y))(ys) / N

    expected_x_probas = np.zeros((T + 1, d_x))
    expected_y_probas = np.zeros((T, d_y))
    expected_x_probas[0] = pi_0

    for t in range(T):
        expected_x_probas[t + 1] = transition @ expected_x_probas[t]
        expected_y_probas[t] = emission @ expected_x_probas[t + 1]
    chex.assert_tree_all_close(actual_x_probas, expected_x_probas, atol=0.01)
    chex.assert_tree_all_close(actual_y_probas, expected_y_probas, atol=0.01)


def test_sampling_time_varying():
    pass


def test_sampling_time_invariant_and_varying_agree():
    pass
