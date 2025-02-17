"""
Tests for inference in the generalized Gaussian SSM.
"""
import jax.numpy as jnp

from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_smoother, EKFIntegrals, UKFIntegrals
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.inference_ukf import unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import random_nlgssm_args
from dynamax.utils.utils import has_tpu
from functools import partial

if has_tpu():
    allclose = partial(jnp.allclose, atol=1e-1)
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