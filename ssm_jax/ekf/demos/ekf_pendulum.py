# Demo of using EKF to track pendulum angle
# Example taken from Simo Särkkä (2013), “Bayesian Filtering and Smoothing,”
from matplotlib import pyplot as plt

import jax.numpy as jnp

from ssm_jax.nlgssm.demos.simulations import PendulumSimulation
from ssm_jax.plotting import plot_nlgssm_pendulum as plot_pendulum
from ssm_jax.nlgssm.containers import NLGSSMParams
from ssm_jax.ekf.inference import extended_kalman_smoother


def ekf_pendulum():
    # Generate random pendulum data
    pendulum = PendulumSimulation()
    states, obs, time_grid = pendulum.sample()

    # Define parameters for EKF
    ekf_params = NLGSSMParams(
        initial_mean=pendulum.initial_state,
        initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
        dynamics_function=pendulum.dynamics_function,
        dynamics_covariance=pendulum.dynamics_covariance,
        emission_function=pendulum.emission_function,
        emission_covariance=pendulum.emission_covariance,
    )

    # Run extended Kalman smoother
    ekf_posterior = extended_kalman_smoother(ekf_params, obs)

    return states, obs, time_grid, ekf_posterior


def plot_ekf_pendulum(states, obs, grid, ekf_posterior):
    dict_figures = {}
    dict_figures["ekf_pendulum_data"] = plot_pendulum(grid, states[:, 0], obs)
    dict_figures["ekf_pendulum_filtered"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ekf_posterior.filtered_means[:, 0], est_type="EKF"
    )
    dict_figures["ekf_pendulum_smoothed"] = plot_pendulum(
        grid, states[:, 0], obs, x_est=ekf_posterior.smoothed_means[:, 0], est_type="EKS"
    )
    return dict_figures


def main(test_mode=False):
    figures = plot_ekf_pendulum(*(ekf_pendulum()))
    return figures

if __name__ == "__main__":
    figures = main()
    plt.show()

