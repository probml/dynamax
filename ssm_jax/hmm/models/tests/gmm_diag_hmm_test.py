import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.models import GaussianMixtureDiagHMM
from ssm_jax.utils import monotonically_increasing


def test_random_initialization(key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2):
    hmm = GaussianMixtureDiagHMM.random_initialization(key, num_states, num_mix, emission_dim)

    assert hmm.initial_probs.value.shape == (num_states,)
    assert jnp.allclose(hmm.initial_probs.value.sum(), 1.)

    assert hmm.transition_matrix.value.shape == (num_states, num_states)
    assert jnp.allclose(hmm.transition_matrix.value.sum(axis=-1), 1.)

    assert hmm.emission_mixture_weights.value.shape == (num_states, num_mix)
    assert jnp.allclose(hmm.emission_mixture_weights.value.sum(axis=-1), 1.)

    assert hmm.emission_means.value.shape == (num_states, num_mix, emission_dim)

    assert hmm.emission_cov_diag_factors.value.shape == (num_states, num_mix, emission_dim)


def test_sample(key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2, num_timesteps=100):
    init_key, sample_key = jr.split(key)
    h = GaussianMixtureDiagHMM.random_initialization(init_key, num_states, num_mix, emission_dim)

    states, emissions = h.sample(sample_key, num_timesteps)
    assert emissions.shape == (num_timesteps, emission_dim)
    assert len(states) == num_timesteps


def test_fit(key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2, num_timesteps=100):
    key0, key1, key2 = jr.split(key, 3)
    true_hmm = GaussianMixtureDiagHMM.random_initialization(key0, num_states, num_mix, emission_dim)
    _, emissions = true_hmm.sample(key2, num_timesteps)

    # Mess up the parameters and see if we can re-learn them.
    hmm = GaussianMixtureDiagHMM.random_initialization(key1, num_states, num_mix, emission_dim)
    lps = hmm.fit_em(emissions[None, ...])
    assert monotonically_increasing(lps, atol=1e-1)
