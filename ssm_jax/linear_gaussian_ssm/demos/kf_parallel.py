# This example demonstrates the use of the functional inference interface in
#  ssm_jax where `jax.vmap` is used to map smoothing over multiple samples.
#
#  Note the use of `lgssm_smoother()` rather than `LinearGaussianSSM.smoother()`

from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from matplotlib import pyplot as plt

from functools import partial

from ssm_jax.plotting import plot_lgssm_posterior
from ssm_jax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM
from ssm_jax.linear_gaussian_ssm.inference import lgssm_smoother


def kf_parallel():
    delta = 1.0
    F = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])

    H = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    state_size, _ = F.shape
    observation_size, _ = H.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 1.0

    # Prior parameter distribution
    mu0 = jnp.array([8, 10, 1, 0]).astype(float)
    Sigma0 = jnp.eye(state_size) * 0.1

    lgssm = LinearGaussianSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=H,
        emission_covariance=R,
    )

    num_timesteps = 15
    num_samples = 4
    key = jr.PRNGKey(310)
    keys = jr.split(key, num_samples)

    xs, ys = vmap(lambda key: lgssm.sample(key, num_timesteps))(keys)

    lgssm_posteriors = vmap(partial(lgssm_smoother, lgssm.params))(ys)

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
