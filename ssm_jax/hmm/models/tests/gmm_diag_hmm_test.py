import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from ssm_jax.hmm.models import GaussianMixtureDiagHMM
from ssm_jax.hmm.models.gmm_hmm import GaussianMixtureHMM
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
    hmm = GaussianMixtureDiagHMM.random_initialization(init_key, num_states, num_mix, emission_dim)

    states, emissions = hmm.sample(sample_key, num_timesteps)
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


def test_gmm_hmm_vs_gmm_diag_hmm(
        key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2, num_timesteps=100):
    init_key, sample_key = jr.split(key)
    diag_hmm = GaussianMixtureDiagHMM.random_initialization(init_key, num_states, num_mix,
                                                            emission_dim)

    full_hmm = GaussianMixtureHMM(diag_hmm.initial_probs.value,
                                  diag_hmm.transition_matrix.value,
                                  diag_hmm._emission_mixture_weights.value,
                                  diag_hmm.emission_means.value,
                                  vmap(lambda x: vmap(jnp.diag)(x))(
                                      diag_hmm.emission_cov_diag_factors.value),
                                  emission_prior_extra_df=0.,
                                  emission_prior_scale=2.)

    states_diag, emissions_diag = diag_hmm.sample(sample_key, num_timesteps)
    states_full, emissions_full = full_hmm.sample(sample_key, num_timesteps)

    assert jnp.allclose(emissions_full, emissions_diag)
    assert jnp.allclose(states_full, states_diag)

    posterior_diag = diag_hmm.smoother(emissions_diag)
    posterior_full = full_hmm.smoother(emissions_diag)

    assert jnp.allclose(posterior_diag.marginal_loglik, posterior_full.marginal_loglik)
    assert jnp.allclose(posterior_diag.filtered_probs, posterior_full.filtered_probs)
    assert jnp.allclose(posterior_diag.predicted_probs, posterior_full.predicted_probs)
    assert jnp.allclose(posterior_diag.smoothed_probs, posterior_full.smoothed_probs)
    assert jnp.allclose(posterior_diag.initial_probs, posterior_full.initial_probs)

    states_diag = diag_hmm.most_likely_states(emissions_diag)
    states_full = full_hmm.most_likely_states(emissions_diag)

    assert jnp.allclose(states_full, states_diag)
