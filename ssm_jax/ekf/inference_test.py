import jax.random as jr
import jax.numpy as jnp

from ssm_jax.lgssm.inference import lgssm_filter, lgssm_smoother
from ssm_jax.ekf.inference import extended_kalman_filter, extended_kalman_smoother
from ssm_jax.nlgssm.sarkka_lib import ekf, eks
from ssm_jax.nlgssm.inference_test import lgssm_to_nlgssm, random_args


# Helper functions
_all_close = lambda x, y: jnp.allclose(x, y, rtol=1e-3)


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    lgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=True)

    # Run standard Kalman filter
    kf_post = lgssm_filter(lgssm, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(lgssm), emissions)

    # Compare filter results
    assert _all_close(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert _all_close(kf_post.filtered_means, ekf_post.filtered_means)
    assert _all_close(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=0, num_timesteps=15):
    nlgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=False)

    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*(nlgssm.return_params), emissions)
    # Run EKF from SSM-Jax
    ekf_post = extended_kalman_filter(nlgssm, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.filtered_means)
    assert _all_close(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    lgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=True)

    # Run standard Kalman smoother
    kf_post = lgssm_smoother(lgssm, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(lgssm_to_nlgssm(lgssm), emissions)

    # Compare smoother results
    assert _all_close(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert _all_close(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def test_extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    nlgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=False)

    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*(nlgssm.return_params), emissions)
    # Run EK smoother from SSM-Jax
    ekf_post = extended_kalman_smoother(nlgssm, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.smoothed_means)
    assert _all_close(covs_ext, ekf_post.smoothed_covariances)
