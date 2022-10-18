# This example demonstrates the use of the functional inference interface in
#  dynamax where `jax.vmap` is used to map smoothing over multiple samples.
#
#  Note the use of `lgssm_smoother()` rather than `LinearGaussianSSM.smoother()`
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from matplotlib import pyplot as plt

from dynamax.plotting import plot_lgssm_posterior
from dynamax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM


def kf_parallel():
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
    num_samples = 4
    key = jr.PRNGKey(310)
    keys = jr.split(key, num_samples)
    xs, ys = vmap(lambda key: lgssm.sample(params, key, num_timesteps))(keys)

    lgssm_posteriors = vmap(lambda y: lgssm.smoother(params, y))(ys)
    return xs, ys, lgssm_posteriors


def plot_kf_parallel(xs, ys, lgssm_posteriors):
    num_samples = len(xs)
    dict_figures = {}

    # Plot Data
    fig, ax = plt.subplots()
    for n in range(num_samples):
        ax.plot(*xs[n, :, :2].T, ls="--", color=f"C{n}")
        ax.plot(*ys[n, ...].T, ".", color=f"C{n}", label=f"Trajectory {n+1}")
    ax.set_title("Data")
    ax.legend()
    dict_figures["missiles_latent"] = fig

    # Plot Filtering
    fig, ax = plt.subplots()
    for n in range(num_samples):
        ax.plot(*ys[n, ...].T, ".")
        filt_means = lgssm_posteriors.filtered_means[n, ...]
        filt_covs = lgssm_posteriors.filtered_covariances[n, ...]
        plot_lgssm_posterior(
            filt_means,
            filt_covs,
            ax,
            color=f"C{n}",
            ellipse_kwargs={"edgecolor": f"C{n}", "linewidth": 0.5},
            label=f"Trajectory {n+1}",
        )
    ax.legend(fontsize=10)
    ax.set_title("Filtered Posterior")
    dict_figures["missiles_filtered"] = fig

    # Plot Smoothing
    fig, ax = plt.subplots()
    for n in range(num_samples):
        ax.plot(*ys[n, ...].T, ".")
        filt_means = lgssm_posteriors.smoothed_means[n, ...]
        filt_covs = lgssm_posteriors.smoothed_covariances[n, ...]
        plot_lgssm_posterior(
            filt_means,
            filt_covs,
            ax,
            color=f"C{n}",
            ellipse_kwargs={"edgecolor": f"C{n}", "linewidth": 0.5},
            label=f"Trajectory {n+1}",
        )
    ax.legend(fontsize=10)
    ax.set_title("Smoothed Posterior")
    dict_figures["missiles_smoothed"] = fig

    return dict_figures


def main(test_mode=False):
    xs, ys, lgssm_posteriors = kf_parallel()
    if not test_mode:
        dict_figures = plot_kf_parallel(xs, ys, lgssm_posteriors)
        plt.show()


if __name__ == "__main__":
    main()
