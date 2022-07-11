import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import nn
from ssm_jax.hmm.models.categorical_hmm import CategoricalHMM
from ssm_jax.hmm.models.categorical_logit_hmm import CategoricalLogitHMM


def init_categorical_hmm_from(categorical_logit_hmm):

    initial_probs = categorical_logit_hmm.initial_probabilities
    emission_probs = categorical_logit_hmm.emission_probs
    transition_matrix = categorical_logit_hmm.transition_matrix

    categorical_hmm = CategoricalHMM(initial_probs, transition_matrix, emission_probs)
    return categorical_hmm


def test_categorical_hmm_parameters(key=jr.PRNGKey(0), num_states=2, emission_dim=2):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    emission_probs = categorical_logit_hmm.emission_probs
    transition_matrix = categorical_logit_hmm.transition_matrix

    assert categorical_hmm.transition_matrix.shape == transition_matrix.shape
    assert categorical_hmm.emission_probs.shape == emission_probs.shape

    assert jnp.allclose(categorical_hmm.transition_matrix, transition_matrix)
    assert jnp.allclose(categorical_hmm.emission_probs, emission_probs)

    assert jnp.allclose(categorical_hmm.emission_probs, nn.softmax(categorical_logit_hmm.emission_logits, axis=-1))
    assert jnp.allclose(categorical_hmm.transition_matrix, nn.softmax(categorical_logit_hmm.transition_logits, axis=-1))


def test_categorical_hmm_sample(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    states, emissions = categorical_logit_hmm.sample(key, num_timesteps)
    test_states, test_emissions = categorical_hmm.sample(key, num_timesteps)

    assert jnp.allclose(states, test_states)
    assert jnp.allclose(emissions, test_emissions)


def test_categorical_hmm_log_prob(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    states, emissions = categorical_logit_hmm.sample(key, num_timesteps)
    assert jnp.allclose(categorical_hmm.log_prob(states, emissions), categorical_logit_hmm.log_prob(states, emissions))


def test_categorical_hmm_conditional_logliks(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    _, emissions = categorical_logit_hmm.sample(key, num_timesteps)
    assert jnp.allclose(
        categorical_hmm._conditional_logliks(emissions),
        categorical_logit_hmm._conditional_logliks(emissions),
        atol=1e-3,
    )


def test_categorical_hmm_marginal_log_prob(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    _, emissions = categorical_logit_hmm.sample(key, num_timesteps)
    assert jnp.allclose(
        categorical_hmm.marginal_log_prob(emissions), categorical_logit_hmm.marginal_log_prob(emissions)
    )


def test_categorical_hmm_most_likely_states(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    _, emissions = categorical_logit_hmm.sample(key, num_timesteps)
    assert jnp.allclose(
        categorical_hmm.most_likely_states(emissions), categorical_logit_hmm.most_likely_states(emissions)
    )


def test_categorical_hmm_filter(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    _, emissions = categorical_logit_hmm.sample(key, num_timesteps)

    posterior = categorical_logit_hmm.smoother(emissions)
    test_posterior = categorical_hmm.smoother(emissions)

    assert jnp.allclose(posterior.marginal_loglik, test_posterior.marginal_loglik)
    assert jnp.allclose(posterior.filtered_probs, test_posterior.filtered_probs, atol=1e-3)
    assert jnp.allclose(posterior.predicted_probs, test_posterior.predicted_probs, atol=1e-3)
    assert jnp.allclose(posterior.smoothed_probs, test_posterior.smoothed_probs, atol=1e-3)


def test_categorical_hmm_smoother(key=jr.PRNGKey(0), num_states=4, emission_dim=10, num_timesteps=100):

    categorical_logit_hmm = CategoricalLogitHMM.random_initialization(key, num_states, emission_dim)
    categorical_hmm = init_categorical_hmm_from(categorical_logit_hmm)

    _, emissions = categorical_logit_hmm.sample(key, num_timesteps)

    posterior = categorical_logit_hmm.smoother(emissions)
    test_posterior = categorical_hmm.smoother(emissions)

    assert jnp.allclose(posterior.marginal_loglik, test_posterior.marginal_loglik)
    assert jnp.allclose(posterior.filtered_probs, test_posterior.filtered_probs, atol=1e-3)
    assert jnp.allclose(posterior.predicted_probs, test_posterior.predicted_probs, atol=1e-3)
    assert jnp.allclose(posterior.smoothed_probs, test_posterior.smoothed_probs, atol=1e-3)
