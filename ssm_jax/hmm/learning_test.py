import pytest
import itertools as it
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import ssm_jax.hmm.inference as infer
import ssm_jax.hmm.learning as learn
from ssm_jax.hmm.models import GaussianHMM

def make_rnd_hmm():
    # Set dimensions
    num_states = 5
    emission_dim = 2

    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1]
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    # Make a true HMM
    true_hmm = GaussianHMM(initial_probs,
                        transition_matrix,
                        emission_means, 
                        emission_covs)

    return true_hmm

def make_rnd_model_and_data():
    true_hmm  = make_rnd_hmm()
    num_timesteps = 2000
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)
    return true_hmm, true_states, emissions

def test_loglik():
    true_hmm, true_states, emissions  = make_rnd_model_and_data()
    assert jnp.allclose(true_hmm.log_prob(true_states, emissions), 3149.1013, atol=1e-1)
    assert jnp.allclose(true_hmm.marginal_log_prob(emissions), 3149.1047, atol=1e-1)


def test_hmm_fit_em():
    true_hmm, true_states, emissions  = make_rnd_model_and_data()
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    test_hmm_em, logprobs_em = learn.hmm_fit_em(test_hmm_em, emissions, niter=2)
    assert jnp.allclose(logprobs_em[-1], -3600.2395, atol=1e-1)
    mu = np.array(test_hmm_em.emission_distribution.mean())
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0,0], -0.712, atol=1e-1)


def test_hmm_fit_sgd():
    true_hmm, true_states, emissions  = make_rnd_model_and_data()
    test_hmm_sgd = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    # Quick test: 2 iterations
    optimizer = optax.adam(learning_rate=1e-2)
    test_hmm_sgd, losses = learn.hmm_fit_sgd(GaussianHMM, test_hmm_sgd, emissions, optimizer, niter=2)
    print(losses)
    assert jnp.allclose(losses[-1], 2.852, atol=1e-1)
    mu = np.array(test_hmm_sgd.emission_distribution.mean())
    assert jnp.alltrue(mu.shape == (10, 2))
    assert jnp.allclose(mu[0,0], -1.827, atol=1e-1)
