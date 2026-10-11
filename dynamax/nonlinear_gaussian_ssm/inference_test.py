"""
Tests for the extended and unscented Kalman filters and smoothers.
"""
import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from dynamax.linear_gaussian_ssm import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference_test_utils import assert_sample_moments
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter, extended_kalman_smoother, extended_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm.inference_ukf import unscented_kalman_filter, unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import lgssm_to_nlgssm, random_lgssm_args, random_nlgssm_args, make_affine_timing_case
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ekf, eks
from dynamax.utils.utils import has_tpu
from jax import jit, vmap

if has_tpu():
    # TPU has very poor numerical stability
    allclose = partial(jnp.allclose, atol=1e-3)
else:
    allclose = partial(jnp.allclose, atol=1e-4)


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman filter produces the correct filtered moments
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman filter
    kf_post = lgssm_filter(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(args), emissions)

    # Compare filter results
    assert allclose(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=42, num_timesteps=15):
    """
    Test that the extended Kalman filter produces the correct filtered moments
    by comparing it to the sarkka-jax library.
    """
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*args, emissions)
    # Run EKF from dynamax
    ekf_post = extended_kalman_filter(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.filtered_means)
    assert allclose(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman smoother produces the correct smoothed moments
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman smoother
    kf_post = lgssm_smoother(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(lgssm_to_nlgssm(args), emissions)

    # Compare smoother results
    assert allclose(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert allclose(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman smoother produces the correct smoothed moments
    by comparing it to the sarkka-jax library.
    """
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*args, emissions)
    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.smoothed_means)
    assert allclose(covs_ext, ekf_post.smoothed_covariances)


def test_extended_kalman_sampler_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman sampler produces samples with the correct mean
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)
    new_key = jr.split(jr.PRNGKey(key))[1]

    # Run standard Kalman sampler
    kf_sample = lgssm_posterior_sample(new_key, args, emissions)
    # Run extended Kalman sampler
    ekf_sample = extended_kalman_posterior_sample(new_key, lgssm_to_nlgssm(args), emissions)

    # Compare samples
    assert allclose(kf_sample, ekf_sample)
    
    
def test_extended_kalman_sampler_nonlinear(key=0, num_timesteps=15, sample_size=50000):
    """
    Test that the extended Kalman sampler produces samples with the correct mean.
    """
    # note: empirical covariance needs a large sample_size to converge
    
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)
    
    # Run extended Kalman sampler
    sampler = vmap(extended_kalman_posterior_sample, in_axes=(0,None,None))
    keys = jr.split(jr.PRNGKey(key), sample_size)
    ekf_samples = sampler(keys, args, emissions)

    # Compare sample moments to smoother output
    # Use the posterior variance to compute the variance of the Monte Carlo estimate,
    # and check that the differences are within 6 standard deviations.
    post_variance = vmap(jnp.diag)(ekf_post.smoothed_covariances)
    threshold = 6 * jnp.sqrt(post_variance / sample_size)
    empirical_means = ekf_samples.mean(0)
    assert jnp.all(abs(empirical_means - ekf_post.smoothed_means) < threshold)


@pytest.mark.parametrize(
    "filter_fn,smoother_fn",
    [
        (extended_kalman_filter, extended_kalman_smoother),
        (
            partial(unscented_kalman_filter, hyperparams=UKFHyperParams()),
            partial(unscented_kalman_smoother, hyperparams=UKFHyperParams()),
        ),
    ],
    ids=["extended", "unscented"],
)
@pytest.mark.parametrize("num_timesteps", [1, 3])
def test_kalman_inference_input_indexing(filter_fn, smoother_fn, num_timesteps):
    """Check EKF and UKF input and parameter indexing by comparing filtering, smoothing,
    and prediction results with exact LGSSM results.
    """
    params, emissions, inputs, expected = make_affine_timing_case(num_timesteps)
    filtered = jit(lambda y, u: filter_fn(params, y, inputs=u))(emissions, inputs)
    smoothed = jit(lambda y, u: smoother_fn(params, y, inputs=u))(emissions, inputs)
    for posterior in (filtered, smoothed):
        for field in posterior._fields:
            value = getattr(posterior, field)
            if value is not None:
                np.testing.assert_allclose(value, expected[field], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "filter_fn",
    [extended_kalman_filter, partial(unscented_kalman_filter, hyperparams=UKFHyperParams())],
    ids=["extended", "unscented"],
)
@pytest.mark.parametrize("num_timesteps", [1, 4])
@pytest.mark.parametrize("time_varying_covariance", [False, True])
def test_kalman_prediction_length_without_inputs(filter_fn, num_timesteps, time_varying_covariance):
    """Check EKF and UKF prediction lengths and values without inputs, for static and
    time-varying dynamics covariance.

    Check both full-output and prediction-only requests.
    """
    covariance = jnp.eye(1)
    if time_varying_covariance:
        covariance = jnp.broadcast_to(covariance, (num_timesteps, 1, 1)).at[0].set(jnp.nan)
    params = ParamsNLGSSM(
        initial_mean=jnp.array([2.0]), initial_covariance=jnp.eye(1),
        dynamics_function=lambda state: 0.5 * state, dynamics_covariance=covariance,
        emission_function=lambda state: state, emission_covariance=jnp.eye(1),
    )
    emissions = jnp.array([[0.0], [1.0], [-1.0], [2.0]])[:num_timesteps]
    num_predictions = num_timesteps - int(time_varying_covariance)
    # Scalar Kalman updates followed by m <- m/2, P <- P/4 + 1.
    expected_means = np.array([1/2, 13/34, -51/290, 606/1237])[:num_predictions, None]
    expected_covs = np.array([9/8, 77/68, 657/580, 5605/4948])[:num_predictions, None, None]
    for output_fields in (
        ["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
        ["predicted_means", "predicted_covariances"],
    ):
        posterior = jit(lambda y: filter_fn(params, y, output_fields=output_fields))(emissions)
        if "filtered_means" not in output_fields:
            assert posterior.filtered_means is None
            assert posterior.filtered_covariances is None
        np.testing.assert_allclose(posterior.predicted_means, expected_means, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(posterior.predicted_covariances, expected_covs, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("num_timesteps", [1, 3])
def test_extended_kalman_posterior_sample_input_indexing(num_timesteps):
    """Check input and parameter indexing in EKF posterior sampling
    against the exact LGSSM mean and covariance.
    """
    params, emissions, inputs, expected = make_affine_timing_case(num_timesteps)
    samples = jit(vmap(lambda key: extended_kalman_posterior_sample(key, params, emissions, inputs)))(
        jr.split(jr.PRNGKey(31), 10000)
    )
    assert_sample_moments(samples, expected["smoothed_means"], expected["joint_covariance"])


def test_kalman_inference_rejects_short_dynamics_covariance():
    """Check that the EKF and UKF filters and the EKF smoother
    reject incorrect dynamics covariance lengths.
    """
    params, emissions, inputs, _ = make_affine_timing_case()
    filtered = extended_kalman_filter(params, emissions, inputs=inputs)
    params = params._replace(dynamics_covariance=params.dynamics_covariance[:-1])
    for inference in (
        lambda y, u: extended_kalman_filter(params, y, inputs=u),
        lambda y, u: unscented_kalman_filter(params, y, UKFHyperParams(), inputs=u),
        lambda y, u: extended_kalman_smoother(params, y, filtered_posterior=filtered, inputs=u),
    ):
        with pytest.raises(AssertionError):
            inference(emissions, inputs)
