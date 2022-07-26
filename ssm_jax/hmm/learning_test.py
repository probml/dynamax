import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
import ssm_jax.hmm.learning as learn
from ssm_jax.hmm.models import GaussianHMM


def make_rnd_hmm(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    # Make a true HMM
    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

    return true_hmm


def make_rnd_model_and_data(num_states=5, emission_dim=2, num_timesteps=2000):
    true_hmm = make_rnd_hmm(num_states, emission_dim)
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)
    batch_emissions = emissions[None, ...]
    return true_hmm, true_states, batch_emissions


def test_loglik():
    true_hmm, true_states, batch_emissions = make_rnd_model_and_data()
    assert jnp.allclose(true_hmm.log_prob(true_states, batch_emissions[0]), 3149.1013, atol=1e-1)
    assert jnp.allclose(true_hmm.marginal_log_prob(batch_emissions[0]), 3149.1047, atol=1e-1)


def test_hmm_fit_em(num_iters=2):
    true_hmm, _, batch_emissions = make_rnd_model_and_data()
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    test_hmm_em, logprobs_em = learn.hmm_fit_em(test_hmm_em, batch_emissions, num_iters=num_iters)
    assert jnp.allclose(logprobs_em[-1], -3600.2395, atol=1e-1)
    mu = test_hmm_em.emission_means.value
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0, 0], -0.712, atol=1e-1)


def test_hmm_fit_sgd(num_iters=2):
    true_hmm, _, batch_emissions = make_rnd_model_and_data()
    print(batch_emissions.shape)
    test_hmm_sgd = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    optimizer = optax.adam(learning_rate=1e-2)
    test_hmm_sgd, losses = learn.hmm_fit_sgd(test_hmm_sgd, batch_emissions, optimizer=optimizer, num_iters=num_iters)
    assert jnp.allclose(losses[-1], 2.852, atol=1e-1)
    mu = test_hmm_sgd.emission_means.value
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0, 0], -1.827, atol=1e-1)
