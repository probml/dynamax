import pytest

import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.models import MultivariateNormalDiagPlusLowRankHMM as LowRankHMM


def test_sample_lowrank_hmm(key=jr.PRNGKey(0), num_states=4, emission_dim=2, num_timesteps=100):
    k1, k2 = jr.split(key, 2)

    hmm = LowRankHMM.random_initialization(k1, num_states, emission_dim)
    hmm.emission_covariance_matrices.value *= 0.1
    states, emissions = hmm.sample(k2, num_timesteps)


def test_lowrank_hmm_em(key=jr.PRNGKey(0), num_states=4, emission_dim=2,  num_timesteps=100):
    k1, k2, k3 = jr.split(key, 3)

    hmm = LowRankHMM.random_initialization(k1, num_states, emission_dim)
    hmm.emission_covariance_matrices.value *= 0.1
    states, emissions = hmm.sample(k2, num_timesteps)

    # Try fitting it!
    test_hmm = LowRankHMM.random_initialization(k3, num_states, emission_dim)
    lps = test_hmm.fit_em(jnp.expand_dims(emissions, 0), num_iters=3)
    assert jnp.all(jnp.diff(lps) >= -1e-3)


def test_sample_lowrank_viterbi(key=jr.PRNGKey(0), num_states=4, emission_dim=2,  num_timesteps=100):
    k1, k2 = jr.split(key, 2)

    hmm = LowRankHMM.random_initialization(k1, num_states, emission_dim)
    hmm.emission_covariance_matrices.value *= 0.1
    states, emissions = hmm.sample(k2, num_timesteps)

    # Compute the most likely states
    most_likely_states = hmm.most_likely_states(emissions)
