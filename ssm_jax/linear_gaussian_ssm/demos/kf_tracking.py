# This demo provides a basic example of Kalman filtering and
#  smoothing with ssm_jax.
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from ssm_jax.plotting import plot_lgssm_posterior
from ssm_jax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM


def kf_tracking():
    state_dim = 4
    emission_dim = 2
    delta = 1.0

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.random_initialization(jr.PRNGKey(0))
    params['initial']['mean'] = jnp.array([8.0, 10.0, 1.0, 0.0])
    params['initial']['cov'] = jnp.eye(state_dim) * 0.1
    params['dynamics']['weights'] = jnp.array([[1, 0, delta, 0],
                                               [0, 1, 0, delta],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]])
    params['dynamics']['cov'] = jnp.eye(state_dim) * 0.001
    params['emissions']['weights'] = jnp.array([[1.0, 0, 0, 0],
                                                [0, 1.0, 0, 0]])
    params['emissions']['cov'] = jnp.eye(emission_dim) * 1.0

    num_timesteps = 15
    key = jr.PRNGKey(310)
    x, y = lgssm.sample(params, key, num_timesteps)
    lgssm_posterior = lgssm.smoother(params, y)
    return x, y, lgssm_posterior


def plot_kf_tracking(x, y, lgssm_posterior):

    observation_marker_kwargs = {"marker": "o", "markerfacecolor": "none", "markeredgewidth": 2, "markersize": 8}
    dict_figures = {}

    # Plot Data
    fig1, ax1 = plt.subplots()
    ax1.plot(*x[:, :2].T, marker="s", color="C0", label="true state")
    ax1.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="emissions")
    ax1.legend(loc="upper left")

    # Plot Filtering
    fig2, ax2 = plt.subplots()
    ax2.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
    ax2.plot(*x[:, :2].T, ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.filtered_means,
        lgssm_posterior.filtered_covariances,
        ax2,
        color="tab:red",
        label="filtered means",
        ellipse_kwargs={"edgecolor": "k", "linewidth": 0.5},
        legend_kwargs={"loc":"upper left"}
    )

    # Plot Smoothing
    fig3, ax3 = plt.subplots()
    ax3.plot(*y.T, ls="", **observation_marker_kwargs, color="tab:green", label="observed")
    ax3.plot(*x[:, :2].T, ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.smoothed_means,
        lgssm_posterior.smoothed_covariances,
        ax3,
        color="tab:red",
        label="smoothed means",
        ellipse_kwargs={"edgecolor": "k", "linewidth": 0.5},
        legend_kwargs={"loc":"upper left"}
    )

    dict_figures["kalman_tracking_truth"] = fig1
    dict_figures["kalman_tracking_filtered"] = fig2
    dict_figures["kalman_tracking_smoothed"] = fig3

    return dict_figures


def main(test_mode=False):
    x, y, lgssm_posterior = kf_tracking()
    if not test_mode:
        dict_figures = plot_kf_tracking(x, y, lgssm_posterior)
        plt.show()


if __name__ == "__main__":
    main()
