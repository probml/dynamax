# This example demonstrates the use of the lgssm filtering and smoothing when
#  the linear dynamical system induced by the matrix F has imaginary eigenvalues.
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from dynamax.plotting import plot_lgssm_posterior
from dynamax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM


def kf_spiral():
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


def plot_kf_spiral(x, y, lgssm_posterior):
    dict_figures = {}
    fig, ax = plt.subplots()
    ax.plot(*x[:, :2].T, ls="--", color="darkgrey", marker="o", markersize=5, label="true state")
    dict_figures["spiral_data"] = fig

    fig, ax = plt.subplots()
    ax.plot(*x[:, :2].T, ls="--", color="darkgrey", marker="o", markersize=5, label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.filtered_means,
        lgssm_posterior.filtered_covariances,
        ax=ax,
        color="tab:red",
        label="filtered",
        ellipse_kwargs={"linewidth": 0.5},
    )
    dict_figures["spiral_filtered"] = fig

    fig, ax = plt.subplots()
    ax.plot(*x[:, :2].T, ls="--", color="darkgrey", marker="o", markersize=5, label="true state")
    plot_lgssm_posterior(
        lgssm_posterior.smoothed_means,
        lgssm_posterior.smoothed_covariances,
        ax=ax,
        color="tab:red",
        label="smoothed",
        ellipse_kwargs={"linewidth": 0.5},
    )
    dict_figures["spiral_smoothed"] = fig
    return dict_figures


def main(test_mode=False):
    x, y, lgssm_posterior = kf_spiral()
    if not test_mode:
        dict_figures = plot_kf_spiral(x, y, lgssm_posterior)
        plt.show()


if __name__ == "__main__":
    main()
