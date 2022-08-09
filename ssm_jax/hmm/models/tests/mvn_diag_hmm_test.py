import jax.numpy as jnp
import jax.random as jr
import pytest
from ssm_jax.hmm.models.mvn_diag_hmm import MultivariateNormalDiagHMM
from ssm_jax.utils import monotonically_increasing


class TestGaussianHMMWithDiagonalCovars:

    def setup(self):
        key = jr.PRNGKey(42)
        self.num_states = 3
        self.emission_dim = 3
        self.true_hmm = MultivariateNormalDiagHMM.random_initialization(key, self.num_states, self.emission_dim)

    def test_random_initialization(self, key=jr.PRNGKey(0)):
        hmm = MultivariateNormalDiagHMM.random_initialization(key, self.num_states, self.emission_dim)

        assert hmm.initial_probs.value.shape == (self.num_states,)
        assert jnp.allclose(jnp.sum(hmm.initial_probs.value), 1)
        assert hmm.transition_matrix.value.shape == (self.num_states, self.num_states)
        assert jnp.allclose(jnp.sum(hmm.transition_matrix.value, axis=-1), 1)
        assert hmm.emission_means.value.shape == (self.num_states, self.emission_dim)
        assert hmm._emission_cov_diag_factors.value.shape == (self.num_states, self.emission_dim)

    def test_fit(self, key=jr.PRNGKey(0), num_timesteps=100):

        state_sequence, emissions = self.true_hmm.sample(key, num_timesteps)
        hmm = MultivariateNormalDiagHMM.random_initialization(key, self.num_states, self.emission_dim)

        lps = hmm.fit_em(emissions[None, ...])
        assert monotonically_increasing(lps, atol=1)

    def test_filter(self, key=jr.PRNGKey(0), num_timesteps=100):
        state_sequence, emissions = self.true_hmm.sample(key, num_timesteps)

        hmm = MultivariateNormalDiagHMM.random_initialization(key, self.num_states, self.emission_dim)

        posteriors = hmm.filter(emissions)
        assert not jnp.isnan(posteriors.filtered_probs).any()
        assert jnp.allclose(posteriors.filtered_probs.sum(axis=1), 1.)

    def test_smooth(self, key=jr.PRNGKey(0), num_timesteps=100):
        state_sequence, emissions = self.true_hmm.sample(key, num_timesteps)

        hmm = MultivariateNormalDiagHMM.random_initialization(key, self.num_states, self.emission_dim)

        posteriors = hmm.smoother(emissions)
        assert not jnp.isnan(posteriors.filtered_probs).any()
        assert jnp.allclose(posteriors.filtered_probs.sum(axis=1), 1.)

        assert not jnp.isnan(posteriors.smoothed_probs).any()
        assert jnp.allclose(posteriors.smoothed_probs.sum(axis=1), 1.)
