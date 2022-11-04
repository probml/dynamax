import jax.numpy as jnp

from dynamax.nonlinear_gaussian_ssm.generalized_gaussian_filter import EKFParams, UKFParams, general_gaussian_smoother
from dynamax.nonlinear_gaussian_ssm.extended_kalman_filter import extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter import unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.inference_test import random_nlgssm_args


# Helper functions
_all_close = lambda x, y: jnp.allclose(x, y, atol=1e-1)


def test_ekf(key=0, num_timesteps=15):
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from dynamax.ekf
    ekf_post = extended_kalman_smoother(nlgssm_args, emissions)
    # Run EKF as a GGF
    ekf_params = EKFParams(
        initial_mean = nlgssm_args.initial_mean,
        initial_covariance = nlgssm_args.initial_covariance,
        dynamics_function = nlgssm_args.dynamics_function,
        dynamics_covariance = nlgssm_args.dynamics_covariance,
        emission_function = nlgssm_args.emission_function,
        emission_covariance = nlgssm_args.emission_covariance,
    )
    ggf_post = general_gaussian_smoother(ekf_params, emissions)

    # Compare filter and smoother results
    error_ll = jnp.max(ekf_post.marginal_loglik - ggf_post.marginal_loglik)
    error_filtered_means = jnp.max(ekf_post.filtered_means - ggf_post.filtered_means)
    error_filtered_covs = jnp.max(ekf_post.filtered_covariances - ggf_post.filtered_covariances) # not as close
    error_smoothed_means = jnp.max(ekf_post.smoothed_means - ggf_post.smoothed_means)
    error_smoothed_covs = jnp.max(ekf_post.smoothed_covariances - ggf_post.smoothed_covariances)  # not as close
    print([error_ll, error_filtered_means, error_filtered_covs, error_smoothed_means, error_smoothed_covs])

    assert _all_close(ekf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert _all_close(ekf_post.filtered_means, ggf_post.filtered_means)
    assert _all_close(ekf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert _all_close(ekf_post.smoothed_means, ggf_post.smoothed_means)
    assert _all_close(ekf_post.smoothed_covariances, ggf_post.smoothed_covariances)


def test_ukf(key=0, num_timesteps=15):
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)
    hyperparams = UKFHyperParams()

    # Run UKF from dynamax.ukf
    ukf_post = unscented_kalman_smoother(nlgssm_args, emissions, hyperparams)
    # Run UKF as GGF
    ukf_params = UKFParams(
        initial_mean = nlgssm_args.initial_mean,
        initial_covariance = nlgssm_args.initial_covariance,
        dynamics_function = nlgssm_args.dynamics_function,
        dynamics_covariance = nlgssm_args.dynamics_covariance,
        emission_function = nlgssm_args.emission_function,
        emission_covariance = nlgssm_args.emission_covariance,
    )
    ggf_post = general_gaussian_smoother(ukf_params, emissions)

    # Compare filter and smoother results
    assert _all_close(ukf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert _all_close(ukf_post.filtered_means, ggf_post.filtered_means)
    assert _all_close(ukf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert _all_close(ukf_post.smoothed_means, ggf_post.smoothed_means)
    assert _all_close(ukf_post.smoothed_covariances, ggf_post.smoothed_covariances)
