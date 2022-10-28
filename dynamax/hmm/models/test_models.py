import pytest
from datetime import datetime
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import dynamax.hmm.models as models
from dynamax.utils import ensure_array_has_batch_dim, monotonically_increasing

NUM_TIMESTEPS = 100

CONFIGS = [
    (models.BernoulliHMM, dict(num_states=4, emission_dim=3), None),
    (models.CategoricalHMM, dict(num_states=4, num_emissions=3, num_classes=5), None),
    (models.CategoricalRegressionHMM, dict(num_states=4, num_classes=3, feature_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.GaussianHMM, dict(num_states=4, emission_dim=3), None),
    (models.DiagonalGaussianHMM, dict(num_states=4, emission_dim=3), None),
    (models.GaussianMixtureHMM, dict(num_states=4, num_components=2, emission_dim=3), None),
    (models.DiagonalGaussianMixtureHMM, dict(num_states=4, num_components=2, emission_dim=3), None),
    (models.LinearRegressionHMM, dict(num_states=4, emission_dim=3, feature_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.LogisticRegressionHMM, dict(num_states=4, feature_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.MultivariateNormalSphericalHMM, dict(num_states=4, emission_dim=3), None),
    #(models.MultivariateNormalTiedHMM, dict(num_states=4, emission_dim=3), None),
    (models.MultinomialHMM, dict(num_states=4, emission_dim=3, num_classes=5, num_trials=10), None),
    (models.PoissonHMM, dict(num_states=4, emission_dim=3), None),
]


@pytest.mark.parametrize(["cls", "kwargs", "covariates"], CONFIGS)
def test_sample_and_fit(cls, kwargs, covariates):
    hmm = cls(**kwargs)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(42))
    params, param_props = hmm.initialize(key1)
    states, emissions = hmm.sample(params, key2, num_timesteps=NUM_TIMESTEPS, covariates=covariates)
    fitted_params, lps = hmm.fit_em(params, param_props, emissions, covariates=covariates, num_iters=10)
    assert monotonically_increasing(lps, atol=1e-2, rtol=1e-2)
    fitted_params, lps = hmm.fit_sgd(params, param_props, emissions, covariates=covariates, num_epochs=10)


## A few model-specific tests
def test_categorical_hmm_viterbi():
    # From http://en.wikipedia.org/wiki/Viterbi_algorithm:
    hmm = models.CategoricalHMM(num_states=2, num_emissions=1, num_classes=3)
    params, props = hmm.random_initialization(jr.PRNGKey(0))
    params['initial']['probs'] = jnp.array([0.6, 0.4])
    params['transitions']['transition_matrix'] = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    params['emissions']['probs'] = jnp.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]).reshape(2, 1, 3)
    emissions = jnp.arange(3).reshape(3, 1)
    state_sequence = hmm.most_likely_states(params, emissions)
    assert jnp.allclose(jnp.squeeze(state_sequence), jnp.array([1, 0, 0]))


def test_gmm_hmm_vs_gmm_diag_hmm(key=jr.PRNGKey(0), num_states=4, num_components=3, emission_dim=2):
    key1, key2, key3 = jr.split(key, 3)
    diag_hmm = models.DiagonalGaussianMixtureHMM(num_states, num_components, emission_dim)
    diag_params, _ = diag_hmm.random_initialization(key1)
    full_hmm = models.GaussianMixtureHMM(num_states, num_components, emission_dim)
    full_params, _ = full_hmm.random_initialization(key2)

    # Copy over a few params
    full_params['initial']['probs'] = diag_params['initial']['probs']
    full_params['transitions']['transition_matrix'] = diag_params['transitions']['transition_matrix']
    full_params['emissions']['weights'] = diag_params['emissions']['weights']
    full_params['emissions']['means'] = diag_params['emissions']['means']
    full_params['emissions']['covs'] = vmap(lambda ss: vmap(lambda s: jnp.diag(s**2))(ss))(diag_params['emissions']['scale_diags'])

    states_diag, emissions_diag = diag_hmm.sample(diag_params, key3, NUM_TIMESTEPS)
    states_full, emissions_full = full_hmm.sample(full_params, key3, NUM_TIMESTEPS)
    assert jnp.allclose(emissions_full, emissions_diag)
    assert jnp.allclose(states_full, states_diag)

    posterior_diag = diag_hmm.smoother(diag_params, emissions_diag)
    posterior_full = full_hmm.smoother(full_params, emissions_full)

    assert jnp.allclose(posterior_diag.marginal_loglik, posterior_full.marginal_loglik)
    assert jnp.allclose(posterior_diag.filtered_probs, posterior_full.filtered_probs)
    assert jnp.allclose(posterior_diag.predicted_probs, posterior_full.predicted_probs)
    assert jnp.allclose(posterior_diag.smoothed_probs, posterior_full.smoothed_probs)
    assert jnp.allclose(posterior_diag.initial_probs, posterior_full.initial_probs)

    states_diag = diag_hmm.most_likely_states(diag_params, emissions_diag)
    states_full = full_hmm.most_likely_states(full_params, emissions_full)
    assert jnp.allclose(states_full, states_diag)


def test_sample_and_fit_arhmm():
    arhmm = models.LinearAutoregressiveHMM(num_states=4, emission_dim=2, num_lags=1)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(42))
    params, param_props = arhmm.random_initialization(key1)
    states, emissions = arhmm.sample(params, key2, num_timesteps=NUM_TIMESTEPS)
    covariates = arhmm.compute_covariates(emissions)
    fitted_params, lps = arhmm.fit_em(params, param_props, emissions, covariates=covariates, num_iters=10)
    assert monotonically_increasing(lps, atol=1e-2, rtol=1e-2)
    fitted_params, lps = arhmm.fit_sgd(params, param_props, emissions, covariates=covariates, num_epochs=10)


# def test_kmeans_initialization(key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2, num_samples=1000):
#     hmm = GaussianMixtureHMM.random_initialization(key, num_states, num_mix, emission_dim)
#     key0, key1, key2 = jr.split(key, 3)
#     true_hmm = GaussianMixtureHMM.random_initialization(key0, num_states, num_mix, emission_dim)
#     _, emissions = true_hmm.sample(key1, num_samples)
#     hmm = GaussianMixtureHMM.kmeans_initialization(key2, num_states, num_mix, emission_dim, emissions)
#     assert hmm.initial_probs.value.shape == (num_states,)
#     assert jnp.allclose(hmm.initial_probs.value.sum(), 1.)
#     assert hmm.transition_matrix.value.shape == (num_states, num_states)
#     assert jnp.allclose(hmm.transition_matrix.value.sum(axis=-1), 1.)
#     assert hmm.emission_mixture_weights.value.shape == (num_states, num_mix)
#     assert jnp.allclose(hmm.emission_mixture_weights.value.sum(axis=-1), 1.)
#     assert hmm.emission_means.value.shape == (num_states, num_mix, emission_dim)
#     assert jnp.isnan(hmm.emission_means.value).any() == False
#     assert hmm.emission_covariance_matrices.value.shape == (num_states, num_mix, emission_dim, emission_dim)


# def test_kmeans_plusplus_initialization(key=jr.PRNGKey(0), num_states=4, num_mix=3, emission_dim=2, num_samples=1000):
#     hmm = GaussianMixtureHMM.random_initialization(key, num_states, num_mix, emission_dim)
#     key0, key1, key2 = jr.split(key, 3)
#     true_hmm = GaussianMixtureHMM.random_initialization(key0, num_states, num_mix, emission_dim)
#     _, emissions = true_hmm.sample(key1, num_samples)
#     hmm = GaussianMixtureHMM.kmeans_plusplus_initialization(key2, num_states, num_mix, emission_dim, emissions)
#     assert hmm.initial_probs.value.shape == (num_states,)
#     assert jnp.allclose(hmm.initial_probs.value.sum(), 1.)
#     assert hmm.transition_matrix.value.shape == (num_states, num_states)
#     assert jnp.allclose(hmm.transition_matrix.value.sum(axis=-1), 1.)
#     assert hmm.emission_mixture_weights.value.shape == (num_states, num_mix)
#     assert jnp.allclose(hmm.emission_mixture_weights.value.sum(axis=-1), 1.)
#     assert hmm.emission_means.value.shape == (num_states, num_mix, emission_dim)
#     assert jnp.isnan(hmm.emission_means.value).any() == False
#     assert hmm.emission_covariance_matrices.value.shape == (num_states, num_mix, emission_dim, emission_dim)


# @pytest.mark.skip(reason="this would introduce a torch dependency")
# def test_hmm_fit_stochastic_em(num_iters=100):
#     """Evaluate stochastic em fit with respect to exact em fit."""

#     # -------------------------------------------------------------
#     def _collate(batch):
#         """Merges a list of samples to form a batch of tensors."""
#         if isinstance(batch[0], jnp.ndarray):
#             return jnp.stack(batch)
#         elif isinstance(batch[0], (tuple,list)):
#             transposed = zip(*batch)
#             return [_collate(samples) for samples in transposed]
#         else:
#             return jnp.array(batch)


#     from torch.utils.data import DataLoader
#     class ArrayLoader(DataLoader):
#         """Generates an iterable over the given array, with option to reshuffle.

#         Args:
#             dataset (chex.Array or Dataset): Any object implementing __len__ and __getitem__
#             batch_size (int): Number of samples to load per batch
#             shuffle (bool): If True, reshuffle data at every epoch
#             drop_last (bool): If true, drop last incomplete batch if dataset size is
#                 not divisible by batch size, drop last incomplete batch. Else, keep
#                 (smaller) last batch.
#         """
#         def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=True):

#             super(self.__class__, self).__init__(dataset,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 collate_fn=_collate,
#                 drop_last=drop_last
#                 )
#     # Generate data and construct dataloader
#     true_hmm, _, batch_emissions = make_rnd_model_and_data(num_batches=8)
#     emissions_generator = ArrayLoader(batch_emissions, batch_size=2, shuffle=True)

#     refr_hmm = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
#     test_hmm = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)

#     refr_lps = refr_hmm.fit_em(batch_emissions, num_iters)

#     total_emissions = len(batch_emissions.reshape(-1, true_hmm.num_obs))
#     test_lps = test_hmm.fit_stochastic_em(
#         emissions_generator, total_emissions, num_epochs=num_iters,
#     )

#     # -------------------------------------------------------------------------
#     # we expect lps to likely differ by quite a bit, but should be in the same order
#     print(f'test log prob {test_lps[-1]:.2f} refrence lp {refr_lps[-1]:.2f}')
#     assert jnp.allclose(test_lps[-1], refr_lps[-1], atol=100)

#     refr_mu = refr_hmm.emission_means.value
#     test_mu = test_hmm.emission_means.value

#     assert jnp.alltrue(test_mu.shape == (10, 2))
#     assert jnp.allclose(jnp.linalg.norm(test_mu-refr_mu, axis=-1), 0., atol=2)

#     refr_cov = refr_hmm.emission_covariance_matrices.value
#     test_cov = test_hmm.emission_covariance_matrices.value
#     assert jnp.alltrue(test_cov.shape == (10, 2, 2))
#     assert jnp.allclose(jnp.linalg.norm(test_cov-refr_cov, axis=-1), 0., atol=1)
