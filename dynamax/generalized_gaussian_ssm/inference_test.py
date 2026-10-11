"""
Tests for inference in the generalized Gaussian SSM.
"""
import numpy as np
import pytest
import jax.numpy as jnp
from jax import jit

from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, conditional_moments_gaussian_smoother, EKFIntegrals, UKFIntegrals, GHKFIntegrals
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.inference_ukf import unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import random_nlgssm_args, make_affine_timing_case
from dynamax.utils.utils import has_tpu
from functools import partial

if has_tpu():
    allclose = partial(jnp.allclose, atol=1e-3)
else:
    allclose = partial(jnp.allclose, atol=1e-3)


def ekf(key=0, num_timesteps=15):
    """
    Test EKF as a GGF
    """
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from dynamax.ekf
    ekf_post = extended_kalman_smoother(nlgssm_args, emissions)
    # Run EKF as a GGF
    ekf_params = ParamsGGSSM(
        initial_mean=nlgssm_args.initial_mean,
        initial_covariance=nlgssm_args.initial_covariance,
        dynamics_function=nlgssm_args.dynamics_function,
        dynamics_covariance=nlgssm_args.dynamics_covariance,
        emission_mean_function=nlgssm_args.emission_function,
        emission_cov_function=lambda x: nlgssm_args.emission_covariance,
    )
    ggf_post = conditional_moments_gaussian_smoother(ekf_params, EKFIntegrals(), emissions)

    # Compare filter and smoother results
    assert allclose(ekf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert allclose(ekf_post.filtered_means, ggf_post.filtered_means)
    assert allclose(ekf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert allclose(ekf_post.smoothed_means, ggf_post.smoothed_means)
    assert allclose(ekf_post.smoothed_covariances, ggf_post.smoothed_covariances)


def test_ukf(key=1, num_timesteps=15):
    """
    Test UKF as a GGF
    """
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)
    hyperparams = UKFHyperParams()

    # Run UKF from dynamax.ukf
    ukf_post = unscented_kalman_smoother(nlgssm_args, emissions, hyperparams)
    # Run UKF as GGF
    ukf_params = ParamsGGSSM(
        initial_mean=nlgssm_args.initial_mean,
        initial_covariance=nlgssm_args.initial_covariance,
        dynamics_function=nlgssm_args.dynamics_function,
        dynamics_covariance=nlgssm_args.dynamics_covariance,
        emission_mean_function=nlgssm_args.emission_function,
        emission_cov_function=lambda x: nlgssm_args.emission_covariance,
    )
    ggf_post = conditional_moments_gaussian_smoother(ukf_params, UKFIntegrals(), emissions)

    # Compare filter and smoother results
    # c1, c2 = ukf_post.filtered_covariances, ggf_post.filtered_covariances
    # print(c1[0], '\n\n', c2[0])
    assert allclose(ukf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert allclose(ukf_post.filtered_means, ggf_post.filtered_means)
    assert allclose(ukf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert allclose(ukf_post.smoothed_means, ggf_post.smoothed_means)
    assert allclose(ukf_post.smoothed_covariances, ggf_post.smoothed_covariances)

@pytest.mark.parametrize("num_timesteps", [1, 3])
@pytest.mark.parametrize("integrals", [EKFIntegrals(), UKFIntegrals(), GHKFIntegrals(order=3)])
def test_generalized_gaussian_inference_input_indexing(num_timesteps, integrals):
    """Check GGSSM input and parameter indexing against exact LGSSM filtering and smoothing results.

    Also check that incorrect dynamics covariance lengths are rejected.
    """
    params, emissions, inputs, expected = make_affine_timing_case(num_timesteps)
    generalized_params = ParamsGGSSM(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_function=params.dynamics_function,
        dynamics_covariance=params.dynamics_covariance,
        emission_mean_function=params.emission_function,
        emission_cov_function=lambda state, inpt: params.emission_covariance,
    )
    posterior = jit(lambda y, u: conditional_moments_gaussian_smoother(
        generalized_params, integrals, y, inputs=u
    ))(emissions, inputs)
    for field in posterior._fields:
        value = getattr(posterior, field)
        if value is not None:
            np.testing.assert_allclose(value, expected[field], atol=2e-5, rtol=2e-5)

    # Check dynamics covariance lengths in both filtering and smoothing.
    short_params = generalized_params._replace(dynamics_covariance=params.dynamics_covariance[:-1])
    with pytest.raises(AssertionError):
        conditional_moments_gaussian_filter(short_params, integrals, emissions, inputs=inputs)
    with pytest.raises(AssertionError):
        conditional_moments_gaussian_smoother(
            short_params, integrals, emissions, filtered_posterior=posterior, inputs=inputs)
