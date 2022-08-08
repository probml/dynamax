import jax.numpy as jnp
import jax.random as jr
import pytest
from ssm_jax.hmm.models.multinomial_hmm import MultinomialHMM
from ssm_jax.hmm.models.tests.test_utils import normalized
from ssm_jax.utils import monotonically_increasing


class TestMultinomialHMM:

    def setup(self):
        self.num_states = 2
        self.emission_dim = 1
        self.num_classes = 4
        self.num_trials = 5

    def new_hmm(self):

        initial_probabilities = jnp.array([.6, .4])
        transition_matrix = jnp.array([[.8, .2], [.2, .8]])
        emission_probs = jnp.array([[[.5, .3, .1, .1]], [[.1, .1, .4, .4]]])
        return MultinomialHMM(initial_probabilities, transition_matrix, emission_probs, self.num_trials)

    def test_score_samples(self):
        X = jnp.array([
            [1, 1, 3, 0],
            [3, 1, 1, 0],
            [3, 0, 2, 0],
            [2, 2, 0, 1],
            [2, 2, 0, 1],
            [0, 1, 1, 3],
            [1, 0, 3, 1],
            [2, 0, 1, 2],
            [0, 2, 1, 2],
            [1, 0, 1, 3],
        ])
        n_samples = X.shape[0]
        hmm = self.new_hmm()

        posteriors = hmm.filter(X)
        assert posteriors.filtered_probs.shape == (n_samples, self.num_states)
        assert jnp.allclose(posteriors.filtered_probs.sum(axis=1), jnp.ones(n_samples))

    def test_sample(self, key=jr.PRNGKey(0), num_timesteps=1000):
        hmm = self.new_hmm()
        state_sequence, emissions = hmm.sample(key, num_timesteps)
        assert emissions.ndim == 3
        assert len(emissions) == len(state_sequence) == num_timesteps
        assert len(jnp.unique(emissions)) == self.num_trials + 1
        assert (emissions.sum(axis=-1) == self.num_trials).all()

    def test_fit(self, key=jr.PRNGKey(0), num_timesteps=100):

        sample_key, init_key = jr.split(key, 2)
        true_hmm = self.new_hmm()
        state_sequence, emissions = true_hmm.sample(sample_key, num_timesteps)

        # Mess up the parameters and see if we can re-learn them.
        hmm = MultinomialHMM.random_initialization(init_key, self.num_states, self.emission_dim, self.num_classes)

        lps = hmm.fit_em(emissions[None, ...])

        assert monotonically_increasing(lps, 1)

    def test_fit_emission_probs(self, key=jr.PRNGKey(0), num_timesteps=100):

        sample_key, init_key = jr.split(key, 2)
        true_hmm = self.new_hmm()
        state_sequence, emissions = true_hmm.sample(sample_key, num_timesteps)

        # Mess up the parameters and see if we can re-learn them.
        hmm = MultinomialHMM.random_initialization(init_key, self.num_states, self.emission_dim, self.num_classes)
        lps = hmm.fit_em(emissions[None, ...])
        """assert jnp.allclose(prev_initial_probs, hmm.initial_probs.value)
        assert jnp.allclose(prev_transition_matrix, hmm.transition_matrix.value)"""
        assert monotonically_increasing(lps, 1)
