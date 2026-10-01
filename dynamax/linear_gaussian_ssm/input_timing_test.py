"""Tests for input and time-varying parameter indexing in Gaussian inference."""

import numpy as np
import pytest
from jax import jit, random as jr, vmap

from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_posterior_sample, lgssm_smoother
from dynamax.linear_gaussian_ssm.inference_test_utils import make_timing_case, assert_sample_moments


@pytest.mark.parametrize("num_timesteps", [1, 3])
@pytest.mark.parametrize("smoother", [lgssm_smoother])
def test_inference_input_indexing(num_timesteps, smoother):
    """Filtering and smoothing match the hand-computed Gaussian posterior."""
    params, emissions, inputs, expected = make_timing_case(num_timesteps)
    actual = jit(smoother)(params, emissions, inputs)._asdict()
    for field, value in actual.items():
        if value is not None:
            np.testing.assert_allclose(value, expected[field], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("num_timesteps", [1, 3])
@pytest.mark.parametrize("sampler", [lgssm_posterior_sample])
def test_posterior_sample_input_indexing(num_timesteps, sampler):
    """Posterior samples match the expected mean and covariance."""
    params, emissions, inputs, expected = make_timing_case(num_timesteps)
    samples = jit(vmap(lambda key: sampler(key, params, emissions, inputs)))(
        jr.split(jr.PRNGKey(13), 12000)
    )
    assert_sample_moments(samples, expected["smoothed_means"], expected["joint_covariance"])


@pytest.mark.parametrize("filter_fn", [lgssm_filter])
def test_rejects_short_dynamics_covariance(filter_fn):
    """A missing covariance must raise instead of reusing the final entry."""
    params, emissions, inputs, _ = make_timing_case()
    params = params._replace(dynamics=params.dynamics._replace(cov=params.dynamics.cov[:-1]))
    with pytest.raises(AssertionError):
        filter_fn(params, emissions, inputs)
