"""Tests for LGSSM input and parameter indexing and covariance lengths."""

import numpy as np
import pytest
import jax.numpy as jnp
from jax import jit, random as jr, vmap

from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_posterior_sample, lgssm_smoother
from dynamax.linear_gaussian_ssm import parallel_lgssm_filter, parallel_lgssm_posterior_sample, parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.info_inference import ParamsLGSSMInfo, info_to_moment_form, lgssm_info_filter, lgssm_info_smoother
from dynamax.linear_gaussian_ssm.inference_test_utils import make_timing_case, assert_sample_moments


def _to_info_form(params):
    """Express the same time-varying model using precisions instead of covariances."""
    return ParamsLGSSMInfo(
        initial_mean=params.initial.mean,
        initial_precision=jnp.linalg.inv(params.initial.cov),
        dynamics_weights=params.dynamics.weights,
        dynamics_precision=jnp.linalg.inv(params.dynamics.cov),
        dynamics_input_weights=params.dynamics.input_weights,
        dynamics_bias=params.dynamics.bias,
        emission_weights=params.emissions.weights,
        emission_precision=jnp.linalg.inv(params.emissions.cov),
        emission_input_weights=params.emissions.input_weights,
        emission_bias=params.emissions.bias,
    )


@pytest.mark.parametrize("num_timesteps", [1, 3])
@pytest.mark.parametrize("smoother", [lgssm_smoother, parallel_lgssm_smoother, lgssm_info_smoother])
def test_inference_input_indexing(num_timesteps, smoother):
    """Check input and parameter indexing in LGSSM filtering and smoothing against exact results."""
    params, emissions, inputs, expected = make_timing_case(num_timesteps)
    if smoother == lgssm_info_smoother:
        params = _to_info_form(params)
    actual = jit(smoother)(params, emissions, inputs)._asdict()
    if smoother == lgssm_info_smoother:
        for field in ("filtered", "smoothed"):
            actual[f"{field}_means"], actual[f"{field}_covariances"] = info_to_moment_form(
                actual.pop(f"{field}_etas"), actual.pop(f"{field}_precisions"))
    for field, value in actual.items():
        if value is not None:
            np.testing.assert_allclose(value, expected[field], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("num_timesteps", [1, 3])
@pytest.mark.parametrize("sampler", [lgssm_posterior_sample, parallel_lgssm_posterior_sample])
def test_posterior_sample_input_indexing(num_timesteps, sampler):
    """Check input and parameter indexing in LGSSM posterior sampling
    against the exact mean and covariance.
    """
    params, emissions, inputs, expected = make_timing_case(num_timesteps)
    samples = jit(vmap(lambda key: sampler(key, params, emissions, inputs)))(
        jr.split(jr.PRNGKey(13), 12000)
    )
    assert_sample_moments(samples, expected["smoothed_means"], expected["joint_covariance"])


@pytest.mark.parametrize("filter_fn", [lgssm_filter, parallel_lgssm_filter, lgssm_info_filter])
def test_rejects_short_dynamics_covariance(filter_fn):
    """Check that LGSSM filters reject incorrect dynamics covariance lengths."""
    params, emissions, inputs, _ = make_timing_case()
    params = params._replace(dynamics=params.dynamics._replace(cov=params.dynamics.cov[:-1]))
    if filter_fn == lgssm_info_filter:
        params = _to_info_form(params)
    with pytest.raises(AssertionError):
        filter_fn(params, emissions, inputs)
