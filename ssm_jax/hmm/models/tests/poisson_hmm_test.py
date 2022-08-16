import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vmap
from ssm_jax.hmm.models import PoissonHMM
from ssm_jax.utils import monotonically_increasing


class TestPoissonHMM:

    num_states = 2
    emission_dim = 3

    def new_hmm(self):
        initial_probabilities = jnp.array([0.6, 0.4])
        transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
        emission_rates = jnp.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
        return PoissonHMM(initial_probabilities, transition_matrix, emission_rates)

    def test_random_initialization(self, key=jr.PRNGKey(0)):
        hmm = PoissonHMM.random_initialization(key, self.num_states, self.emission_dim)
        assert jnp.allclose(hmm.initial_probs.value.sum(), 1.)
        assert jnp.allclose(hmm.transition_matrix.value.sum(axis=-1), 1.)
        assert hmm.emission_rates.value.shape == (self.num_states, self.emission_dim)

    def test_score_samples(self, key=jr.PRNGKey(0), num_timesteps=1000):
        hmm = self.new_hmm()
        states, batch_emissions = hmm.sample(key, num_timesteps)

        assert states.ndim == 1
        assert len(batch_emissions) == len(states) == num_timesteps

        posteriors = hmm.filter(batch_emissions)
        assert posteriors.filtered_probs.shape == (num_timesteps, self.num_states)
        assert jnp.allclose(posteriors.filtered_probs.sum(axis=-1), 1.)

    def test_fit_em(self, key=jr.PRNGKey(0), num_obs=10, num_timesteps=100):

        true_hmm = self.new_hmm()

        init_key, sample_key = jr.split(key)
        keys = jr.split(sample_key, num_obs)
        states, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))(keys, num_timesteps)

        hmm = PoissonHMM.random_initialization(init_key, self.num_states, self.emission_dim)
        lps = hmm.fit_em(batch_emissions, num_iters=5)
        assert monotonically_increasing(lps, atol=1)

    def test_fit_sgd(self, key=jr.PRNGKey(0), num_obs=10, num_timesteps=100):

        true_hmm = self.new_hmm()

        init_key, sample_key = jr.split(key)
        keys = jr.split(sample_key, num_obs)
        states, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))(keys, num_timesteps)

        hmm = PoissonHMM.random_initialization(init_key, self.num_states, self.emission_dim)
        losses = hmm.fit_sgd(batch_emissions, batch_size=num_obs, num_epochs=10)
        assert monotonically_increasing(-losses, atol=1)

    def test_fit_emission_rates(self, key=jr.PRNGKey(0), num_obs=10, num_timesteps=1000):

        init_key, sample_key = jr.split(key)

        true_hmm = self.new_hmm()
        keys = jr.split(sample_key, num_obs)
        state_sequence, batch_emissions = vmap(true_hmm.sample, in_axes=(0, None))(keys, num_timesteps)

        hmm = PoissonHMM.random_initialization(init_key, self.num_states, self.emission_dim)

        initial_probs = jnp.asarray(hmm.initial_probs.value)
        transition_matrix = jnp.asarray(hmm.transition_matrix.value)

        hmm.initial_probs.freeze()
        hmm.transition_matrix.freeze()

        lps = hmm.fit_em(batch_emissions)

        assert jnp.allclose(initial_probs, jnp.asarray(hmm.initial_probs.value))
        assert jnp.allclose(transition_matrix, jnp.asarray(hmm.transition_matrix.value))
        assert monotonically_increasing(lps, atol=1)
