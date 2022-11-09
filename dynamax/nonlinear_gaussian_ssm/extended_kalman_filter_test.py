import jax.numpy as jnp

from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother
from dynamax.nonlinear_gaussian_ssm.extended_kalman_filter import extended_kalman_filter, extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ekf, eks
from dynamax.nonlinear_gaussian_ssm.inference_test import lgssm_to_nlgssm, random_lgssm_args, random_nlgssm_args


# Helper functions
_all_close = lambda x, y: jnp.allclose(x, y, atol=1e-1)


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman filter
    kf_post = lgssm_filter(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(args), emissions)

    # Compare filter results
    assert _all_close(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert _all_close(kf_post.filtered_means, ekf_post.filtered_means)
    assert _all_close(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=42, num_timesteps=15):
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*args, emissions)
    # Run EKF from dynamax
    ekf_post = extended_kalman_filter(args, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.filtered_means)
    assert _all_close(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman smoother
    kf_post = lgssm_smoother(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(lgssm_to_nlgssm(args), emissions)

    # Compare smoother results
    assert _all_close(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert _all_close(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def test_extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*args, emissions)
    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)

    # Compare filter results
    assert _all_close(means_ext, ekf_post.smoothed_means)
    assert _all_close(covs_ext, ekf_post.smoothed_covariances)
