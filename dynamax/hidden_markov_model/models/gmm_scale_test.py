"""Check variance units when initializing diagonal mixture emissions."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dynamax.hidden_markov_model import DiagonalGaussianMixtureHMM
from dynamax.utils.distributions import NormalInverseGamma


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("seed", [0, 31])
def test_diagonal_mixture_prior_uses_standard_deviations(dim, seed):
    """The emission covariance must equal the variance from the declared prior."""
    hmm = DiagonalGaussianMixtureHMM(
        2, 3, dim, emission_prior_shape=4.0, emission_prior_scale=3.0, emission_prior_mean_concentration=2.0
    )
    key = jr.PRNGKey(seed)
    params, _ = hmm.initialize(key)
    emission_key = jr.split(key, 3)[2]
    sample_key = jr.split(emission_key, 3)[1]
    prior = NormalInverseGamma(jnp.asarray(0.0), jnp.asarray(2.0), jnp.asarray(4.0), jnp.asarray(3.0))
    variances, means = prior.sample(seed=sample_key, sample_shape=(2, 3, dim))
    np.testing.assert_array_equal(params.emissions.means, means)
    for state in range(2):
        components = hmm.emission_component.distribution(params.emissions, state).components_distribution
        np.testing.assert_allclose(components.variance(), variances[state], atol=1e-6, rtol=1e-6)


def test_diagonal_mixture_preserves_explicit_scales():
    """User-supplied standard deviations are already in the emission units."""
    hmm = DiagonalGaussianMixtureHMM(2, 3, 2)
    scales = jnp.arange(1.0, 13.0).reshape(2, 3, 2) / 4
    params, _ = hmm.initialize(jr.PRNGKey(0), emission_scale_diags=scales)
    np.testing.assert_array_equal(params.emissions.scale_diags, scales)
