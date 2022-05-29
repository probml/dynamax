# This example demonstrates the use of the lgssm filtering and smoothing when
#  the linear dynamical system induced by the matrix F has imaginary eigenvalues.

from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from ssm_jax.plotting import plot_lgssm_posterior
from ssm_jax.lgssm.models import LinearGaussianSSM

def kf_spiral():
    delta = 1.0
    F = jnp.array([
        [0.1, 1.1, delta, 0],
        [-1, 1, 0, delta],
        [0, 0, 0.1, 0],
        [0, 0, 0, 0.1]
    ])

    H = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    state_size, _ = F.shape
    observation_size, _ = H.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 2.0

    # Prior parameter distribution
    mu0 = jnp.array([1., 1., 1., 0])
    Sigma0 = jnp.eye(state_size) * 0.1

    lgssm = LinearGaussianSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=H,
        emission_covariance=R)

    num_timesteps = 15
    key = jr.PRNGKey(111)
    inputs = jnp.zeros((num_timesteps,0))

    x, y = lgssm.sample(key,num_timesteps)

    lgssm_posterior = lgssm.smoother(y)

    return x, y, lgssm_posterior

def plot_kf_spiral(x, y, lgssm_posterior):
    dict_figures = {}
    fig, ax = plt.subplots()
    ax.plot(*x[:,:2].T,
             ls="--", color="darkgrey",
             marker="o", markersize=5, label="true state")
    dict_figures["spiral_data"] = fig

    fig, ax = plt.subplots()
    ax.plot(*x[:,:2].T,
             ls="--", color="darkgrey",
             marker="o", markersize=5, label="true state")
    plot_lgssm_posterior(lgssm_posterior.filtered_means,
                         lgssm_posterior.filtered_covariances,
                         ax=ax, color="tab:red", label="filtered",
                         ellipse_kwargs={"linewidth":0.5})
    dict_figures["spiral_filtered"] = fig

    fig, ax = plt.subplots()
    ax.plot(*x[:,:2].T,
             ls="--", color="darkgrey",
             marker="o", markersize=5, label="true state")
    plot_lgssm_posterior(lgssm_posterior.smoothed_means,
                         lgssm_posterior.smoothed_covariances,
                         ax=ax,
                         color="tab:red", label="smoothed",
                         ellipse_kwargs={"linewidth":0.5})
    dict_figures["spiral_smoothed"] = fig
    return dict_figures


def main(test_mode = False):
    x, y, lgssm_posterior = kf_spiral()
    if not test_mode:
        dict_figures = plot_kf_spiral(x, y, lgssm_posterior)
        plt.show()

if __name__ == "__main__":
    main()
