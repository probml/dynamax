import jax.numpy as jnp

from ssm_jax.ggssm.inference import general_gaussian_smoother
from ssm_jax.ggssm.containers import EKFParams, UKFParams
from ssm_jax.ekf.inference import extended_kalman_smoother
from ssm_jax.ukf.inference import unscented_kalman_smoother, UKFHyperParams
from ssm_jax.nlgssm.inference_test import random_args


# Helper functions
_all_close = lambda x, y: jnp.allclose(x, y, rtol=1e-3)


def test_ekf(key=0, num_timesteps=15):
    nlgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=False)

    # Run EKF from ssm_jax.ekf
    ekf_post = extended_kalman_smoother(nlgssm, emissions)
    # Run EKF as a GGF
    ekf_params = EKFParams(
        initial_mean = nlgssm.initial_mean,
        initial_covariance = nlgssm.initial_covariance,
        dynamics_function = nlgssm.dynamics_function,
        dynamics_covariance = nlgssm.dynamics_covariance,
        emission_function = nlgssm.emission_function,
        emission_covariance = nlgssm.emission_covariance,
    )
    ggf_post = general_gaussian_smoother(ekf_params, emissions)

    # Compare filter and smoother results
    assert _all_close(ekf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert _all_close(ekf_post.filtered_means, ggf_post.filtered_means)
    assert _all_close(ekf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert _all_close(ekf_post.smoothed_means, ggf_post.smoothed_means)
    assert _all_close(ekf_post.smoothed_covariances, ggf_post.smoothed_covariances)


def test_ukf(key=0, num_timesteps=15):
    nlgssm, _, emissions = random_args(key=key, num_timesteps=num_timesteps, linear=False)
    hyperparams = UKFHyperParams()

    # Run UKF from ssm_jax.ukf
    ukf_post = unscented_kalman_smoother(nlgssm, emissions, hyperparams)
    # Run UKF as GGF
    ukf_params = UKFParams(
        initial_mean = nlgssm.initial_mean,
        initial_covariance = nlgssm.initial_covariance,
        dynamics_function = nlgssm.dynamics_function,
        dynamics_covariance = nlgssm.dynamics_covariance,
        emission_function = nlgssm.emission_function,
        emission_covariance = nlgssm.emission_covariance,
    )
    ggf_post = general_gaussian_smoother(ukf_params, emissions)

    # Compare filter and smoother results
    assert _all_close(ukf_post.marginal_loglik, ggf_post.marginal_loglik)
    assert _all_close(ukf_post.filtered_means, ggf_post.filtered_means)
    assert _all_close(ukf_post.filtered_covariances, ggf_post.filtered_covariances)
    assert _all_close(ukf_post.smoothed_means, ggf_post.smoothed_means)
    assert _all_close(ukf_post.smoothed_covariances, ggf_post.smoothed_covariances)
