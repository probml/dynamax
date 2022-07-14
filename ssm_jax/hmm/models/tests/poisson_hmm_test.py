"""
Based on
https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/tests/test_poisson_hmm.py
"""

import jax.numpy as jnp
import jax.random as jr
import pytest
from ssm_jax.hmm.models import PoissonHMM


class TestPoissonHMM:

    n_components = 2
    n_features = 3

    def new_hmm(self):
        initial_probabilities = jnp.array([0.6, 0.4])
        transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
        emission_rates = jnp.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
        emission_log_rates = jnp.log(emission_rates)
        hmm = PoissonHMM(initial_probabilities, transition_matrix, emission_log_rates)
        return hmm

    def test_hmm_filter(self, key=jr.PRNGKey(0), num_timesteps=1000):
        hmm = self.new_hmm()
        states, emissions = hmm.sample(key, num_timesteps)
        assert emissions.ndim == 2
        assert len(emissions) == len(states) == num_timesteps

        posterior = hmm.filter(emissions)
        assert posterior.filtered_probs.shape == (num_timesteps, self.n_components)
        assert jnp.allclose(posterior.filtered_probs.sum(axis=-1), jnp.ones((num_timesteps,)))
        assert posterior.smoothed_probs is None
