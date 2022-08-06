import pytest

import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.models import LinearRegressionHMM


def test_sample_linreg_hmm(key=jr.PRNGKey(0), num_states=4, emission_dim=2, feature_dim=3, num_timesteps=100):
    k1, k2, k3 = jr.split(key, 3)

    hmm = LinearRegressionHMM.random_initialization(k1, num_states, emission_dim, feature_dim)
    hmm.emission_covariance_matrices.value *= 0.1

    features = jr.normal(k2, (num_timesteps, feature_dim))
    states, emissions = hmm.sample(k3, num_timesteps, features=features)


def test_linreg_hmm_em(key=jr.PRNGKey(0), num_states=4, emission_dim=2, feature_dim=3, num_timesteps=100):
    k1, k2, k3, k4 = jr.split(key, 4)

    hmm = LinearRegressionHMM.random_initialization(k1, num_states, emission_dim, feature_dim)
    hmm.emission_covariance_matrices.value *= 0.1

    features = jr.normal(k2, (num_timesteps, feature_dim))
    states, emissions = hmm.sample(k3, num_timesteps, features=features)

    # Try fitting it!
    test_hmm = LinearRegressionHMM.random_initialization(k4, num_states, emission_dim, feature_dim)
    lps = test_hmm.fit_em(jnp.expand_dims(emissions, 0), features=jnp.expand_dims(features, 0), num_iters=3)
    assert jnp.all(jnp.diff(lps) >= -1e-3)


def test_sample_linreg_viterbi(key=jr.PRNGKey(0), num_states=4, emission_dim=2, feature_dim=3, num_timesteps=100):
    k1, k2, k3 = jr.split(key, 3)

    hmm = LinearRegressionHMM.random_initialization(k1, num_states, emission_dim, feature_dim)
    hmm.emission_covariance_matrices.value *= 0.1

    features = jr.normal(k2, (num_timesteps, feature_dim))
    states, emissions = hmm.sample(k3, num_timesteps, features=features)

    # Compute the most likely states
    most_likely_states = hmm.most_likely_states(emissions, features=features)
