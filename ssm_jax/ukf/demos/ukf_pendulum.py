# Demo of using UKF to track pendulum angle
# Example taken from Simo Särkkä (2013), “Bayesian Filtering and Smoothing,”
from matplotlib import pyplot as plt

import jax.numpy as jnp

from ssm_jax.nlgssm.demos.simulations import PendulumSimulation
from ssm_jax.plotting import plot_nlgssm_pendulum as plot_pendulum
from ssm_jax.nlgssm.containers import NLGSSMParams
from ssm_jax.ukf.inference import unscented_kalman_smoother, UKFHyperParams


def ukf_pendulum():
    # Generate random pendulum data
    pendulum = PendulumSimulation()
    states, obs, time_grid = pendulum.sample()

    # Define parameters for UKF
    ukf_params = NLGSSMParams(
        initial_mean=pendulum.initial_state,
        initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
        dynamics_function=pendulum.dynamics_function,
        dynamics_covariance=pendulum.dynamics_covariance,
        emission_function=pendulum.emission_function,
        emission_covariance=pendulum.emission_covariance,
    )
    ukf_hyperparams = UKFHyperParams()

    # Run extended Kalman smoother
    ukf_posterior = unscented_kalman_smoother(ukf_params, obs, ukf_hyperparams)

    return states, obs, time_grid, ukf_posterior


def plot_ukf_pendulum(states, obs, grid, ukf_posterior):
    dict_figures = {}
    dict_figures["pendulum_data"] = plot_pendulum(grid, states[:, 0], obs)
    dict_figures["pendulum_filtered"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ukf_posterior.filtered_means[:, 0], est_type="UK Filter"
    )
    dict_figures["pendulum_smoothed"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ukf_posterior.smoothed_means[:, 0], est_type="UK Smoother"
    )
    return dict_figures


def main(test_mode=False):
    if not test_mode:
        _ = plot_ukf_pendulum(*(ukf_pendulum()))
        plt.show()


if __name__ == "__main__":
    main()
