import jax.numpy as jnp
from jax import random
from jax import vmap
from sklearn.datasets import make_spd_matrix
from ssm_jax.hmm.models.gaussian_hmm import GaussianHMM


def get_random_gaussian_hmm_params(key, num_states, num_emissions):
    initial_key, transition_key, means_key = random.split(key, 3)
    initial_probabilities = random.uniform(initial_key, shape=(num_states,))
    initial_probabilities = initial_probabilities / initial_probabilities.sum()
    transition_matrix = random.uniform(transition_key, shape=(num_states, num_states))
    transition_matrix = transition_matrix / jnp.sum(initial_probabilities, axis=-1, keepdims=True)
    emission_means = random.randint(means_key, shape=(num_states, num_emissions), minval=-20.,
                                    maxval=20).astype(jnp.float32)
    emission_covars = jnp.array([
        (make_spd_matrix(num_emissions) + 0.1 * jnp.eye(num_emissions)) for _ in range(num_states)
    ])
    return initial_probabilities, transition_matrix, emission_means, emission_covars


def test_fit_means(key=random.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = random.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = random.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    hmm.transition_matrix.freeze()
    hmm.initial_probs.freeze()
    hmm.emission_covariance_matrices.freeze()

    losses = hmm.fit_sgd(batch_emissions)
    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.transition_matrix.value, transition_matrix)
    assert jnp.allclose(hmm.emission_covariance_matrices.value, emission_covars)


def test_fit_covars(key=random.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = random.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = random.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    # TODO: change the params and uncomment the check
    hmm.transition_matrix.freeze()
    hmm.initial_probs.freeze()
    hmm.emission_means.freeze()

    losses = hmm.fit_sgd(batch_emissions)
    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.transition_matrix.value, transition_matrix)
    assert jnp.allclose(hmm.emission_means.value, emission_means)


def test_fit_transition_matrix(key=random.PRNGKey(0), num_states=3, num_emissions=3, num_samples=10):

    init_key, sample_key = random.split(key, 2)
    initial_probabilities, transition_matrix, emission_means, emission_covars = get_random_gaussian_hmm_params(
        init_key, num_states, num_emissions)
    hmm = GaussianHMM(initial_probabilities, transition_matrix, emission_means, emission_covars)

    keys = random.split(sample_key, num_samples)
    batch_states, batch_emissions = vmap(lambda rng: hmm.sample(rng, 10))(keys)

    # Mess up the parameters and see if we can re-learn them.
    # TODO: change the params and uncomment the check
    hmm.initial_probs.freeze()
    hmm.emission_covariance_matrices.freeze()
    hmm.emission_means.freeze()

    losses = hmm.fit_sgd(batch_emissions)
    assert jnp.allclose(hmm.initial_probs.value, initial_probabilities)
    assert jnp.allclose(hmm.emission_means.value, emission_means)
    assert jnp.allclose(hmm.emission_covariance_matrices.value, emission_covars)