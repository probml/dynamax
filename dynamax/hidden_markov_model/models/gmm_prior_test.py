"""Regression checks for Gaussian mixture emission priors."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import dirichlet, invwishart, multivariate_normal

from dynamax.hidden_markov_model import GaussianMixtureHMM


@pytest.mark.parametrize("dim", [2, 3])
def test_correlated_mixture_prior_density(dim):
    """Evaluate the requested correlated prior, including its normalization."""
    scale = np.diag(np.arange(2.0, dim + 2)) + 0.4
    hmm = GaussianMixtureHMM(
        2,
        2,
        dim,
        emission_prior_scale=jnp.asarray(scale),
        emission_prior_extra_df=4.0,
        emission_prior_mean_concentration=2.0,
    )
    covariance = np.eye(dim) + 0.2
    mean = np.linspace(-0.3, 0.4, dim)
    params, _ = hmm.initialize(
        jr.PRNGKey(0),
        initial_probs=jnp.full(2, 0.5),
        transition_matrix=jnp.full((2, 2), 0.5),
        emission_weights=jnp.full((2, 2), 0.5),
        emission_means=jnp.broadcast_to(jnp.asarray(mean), (2, 2, dim)),
        emission_covariances=jnp.broadcast_to(jnp.asarray(covariance), (2, 2, dim, dim)),
    )
    expected = invwishart.logpdf(covariance, df=dim + 4, scale=scale)
    expected += multivariate_normal.logpdf(mean, mean=np.zeros(dim), cov=covariance / 2)
    expected = 4 * expected + 2 * dirichlet.logpdf([0.5, 0.5], [1.1, 1.1])
    np.testing.assert_allclose(hmm.emission_component.log_prior(params.emissions), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale_kind", ["scalar", "diagonal", "correlated"])
def test_mixture_em_uses_requested_prior_scale(dim, scale_kind):
    """Symmetric-mixture EM must recover the weighted normal-inverse-Wishart mode."""
    scale = 2.0 if scale_kind == "scalar" else np.diag(np.arange(2.0, dim + 2))
    if scale_kind == "correlated":
        scale = scale + 0.4
    data = np.array([[0.2, -0.4, 0.8], [1.2, 0.3, -0.1], [-0.8, 1.1, 0.4], [0.7, -0.2, -0.5]])[:, :dim]
    prior_mean = np.linspace(-0.2, 0.3, dim)
    concentration = 0.7
    hmm = GaussianMixtureHMM(
        2,
        2,
        dim,
        emission_prior_scale=jnp.asarray(scale),
        emission_prior_mean=jnp.asarray(prior_mean),
        emission_prior_extra_df=4.0,
        emission_prior_mean_concentration=concentration,
    )
    params, props = hmm.initialize(
        jr.PRNGKey(1),
        initial_probs=jnp.full(2, 0.5),
        transition_matrix=jnp.full((2, 2), 0.5),
        emission_weights=jnp.full((2, 2), 0.5),
        emission_means=jnp.zeros((2, 2, dim)),
        emission_covariances=jnp.broadcast_to(jnp.eye(dim), (2, 2, dim, dim)),
    )
    fitted, _ = hmm.fit_em(params, props, jnp.asarray(data), num_iters=1, verbose=False)
    count = len(data) / 4
    mean = (concentration * prior_mean + data.sum(axis=0) / 4) / (concentration + count)
    prior_scale = scale * np.eye(dim) if np.ndim(scale) == 0 else scale
    posterior_scale = (
        prior_scale
        + data.T @ data / 4
        + concentration * np.outer(prior_mean, prior_mean)
        - (concentration + count) * np.outer(mean, mean)
    )
    covariance = posterior_scale / (dim + 4 + count + dim + 2)
    np.testing.assert_allclose(fitted.emissions.means[0, 0], mean, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(fitted.emissions.covs[0, 0], covariance, atol=1e-6, rtol=1e-5)
