# This demo provides a basic example of Kalman filtering and
#  smoothing with ssm_jax.

import numpy as np
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from ssm_jax.plotting import plot_lgssm_posterior
from ssm_jax.lgssm.models import LinearGaussianSSM

def kf_tracking():
    delta = 1.0
    F = jnp.array([
        [1., 0, delta, 0],
        [0, 1., 0, delta],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    H = jnp.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0]
    ])

    state_size, _ = F.shape
    observation_size, _ = H.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 1.0

    # Prior parameter distribution
    mu0 = jnp.array([8., 10., 1., 0.])
    Sigma0 = jnp.eye(state_size) * 0.1

    lgssm = LinearGaussianSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=H,
        emission_covariance=R)

    # Sample data from model.
    key = jr.PRNGKey(111)
    num_timesteps = 15
    x, y = lgssm.sample(key,num_timesteps)

    # Calculate filtered and smoothed posterior values.
    lgssm_posterior = lgssm.smoother(y)

    return x, y, lgssm_posterior


def plot_kf_tracking(x,y,lgssm_posterior):

    observation_marker_kwargs={
        "marker":"o",
        "markerfacecolor":"none",
        "markeredgewidth":2,
        "markersize":8
    }

    # Plot Data
    fig, ax = plt.subplots()
    ax.plot(*x[:,:2].T, marker="s", color="C0", label="true state");
    ax.plot(*y.T, ls="",
             **observation_marker_kwargs,
             color="tab:green",
             label="emissions")
    ax.legend()

    # Plot Filtering
    fig, ax = plt.subplots()
    ax.plot(*y.T, ls="",
             **observation_marker_kwargs,
             color="tab:green",
             label="observed")
    ax.plot(*x[:,:2].T,
             ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(lgssm_posterior.filtered_means,
                         lgssm_posterior.filtered_covariances,
                         ax,
                         color="tab:red", label="filtered means",
                         ellipse_kwargs={'edgecolor':'k',
                                         'linewidth':0.5});

    # Plot Smoothing
    fig, ax = plt.subplots()
    ax.plot(*y.T, ls="",
             **observation_marker_kwargs,
             color="tab:green",
             label="observed")
    ax.plot(*x[:,:2].T,
             ls="--", color="darkgrey", label="true state")
    plot_lgssm_posterior(lgssm_posterior.smoothed_means,
                         lgssm_posterior.smoothed_covariances,
                         ax,
                         color="tab:red", label="smoothed means",
                         ellipse_kwargs={'edgecolor':'k',
                                         'linewidth':0.5});
    plt.show()

def main():
    x, y, lgssm_posterior = kf_tracking()
    plot_kf_tracking(x, y, lgssm_posterior)

if __name__ == "__main__":
    main()
