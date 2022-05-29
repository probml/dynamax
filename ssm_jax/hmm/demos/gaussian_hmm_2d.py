"""Demo of a simple Gaussian HMM with 2D emissions.
"""
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import optax

from ssm_jax.hmm.models import GaussianHMM

import matplotlib.pyplot as plt
from ssm_jax.plotting import white_to_color_cmap, COLORS, CMAP


def plot_gaussian_hmm(hmm, emissions, states, ttl = "Emission Distributions"):
    lim = .85 * abs(emissions).max()
    XX, YY = jnp.meshgrid(jnp.linspace(-lim, lim, 100),
                          jnp.linspace(-lim, lim, 100))
    grid = jnp.column_stack((XX.ravel(), YY.ravel()))

    lls = hmm.emission_distribution.log_prob(grid[:, None, :])
    plt.figure(figsize=(6, 6))
    for k in range(hmm.num_states):
        plt.contour(XX, YY, jnp.exp(lls[:,k]).reshape(XX.shape),
                    cmap=white_to_color_cmap(COLORS[k]))
        plt.plot(emissions[states==k, 0],
                 emissions[states==k, 1],
                 'o', mfc=COLORS[k], mec='none', ms=3,
                 alpha=.5)

    plt.plot(emissions[:,0], emissions[:,1], '-k', lw=1, alpha=.25)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(ttl)


def plot_gaussian_hmm_emissions(hmm, emissions, states, xlim=None):
    num_timesteps = len(emissions)
    emission_dim = hmm.num_obs

    # Plot the data superimposed on the generating state sequence
    plt.figure()
    lim = 1.05 * abs(emissions).max()
    plt.imshow(states[None,:],
               aspect="auto",
               interpolation="none",
               cmap=CMAP,
               vmin=0, vmax=len(COLORS)-1,
               extent=(0, num_timesteps, -lim, emission_dim*lim),
               alpha=1)

    means = hmm.emission_means[states]
    for d in range(emission_dim):
        plt.plot(emissions[:,d] + lim * d, '-k')
        plt.plot(means[:,d] + lim * d, ':k')

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)

    plt.xlabel("time")
    plt.yticks(lim * jnp.arange(emission_dim),
               ["$x_{}$".format(d+1) for d in range(emission_dim)])

    plt.title("Simulated data from an HMM")
    plt.tight_layout()


def plot_hmm_posterior(true_states, posterior, plot_timesteps=None):
    if plot_timesteps is None: plot_timesteps = len(true_states)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].imshow(true_states[None,:],
                aspect="auto",
                interpolation="none",
                cmap=CMAP,
                vmin=0, vmax=len(COLORS)-1,
                alpha=1)
    axs[0].set_yticks([])
    axs[0].set_title("true states")

    axs[1].imshow(posterior.smoothed_probs.T,
                aspect="auto",
                interpolation="none",
                cmap="Greys",
                vmin=0, vmax=1)
    axs[1].set_ylabel("state")
    axs[1].set_xlabel("time")
    axs[1].set_title("expected states")
    
    plt.xlim(0, plot_timesteps)
    plt.tight_layout()

def make_hmm(num_states = 5, emission_dim = 2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1],
        jnp.zeros((num_states, emission_dim - 2))
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    true_hmm = GaussianHMM(initial_probs,
                           transition_matrix,
                           emission_means,
                           emission_covs)
    return true_hmm

def demo(num_timesteps=2000,
        plot_timesteps=200,
         test_mode=False):

    true_hmm = make_hmm()
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)

    if not test_mode:
        plot_gaussian_hmm(true_hmm, emissions, true_states, "True HMM")
        #nsteps = np.minimum(200, num_timesteps)
        plot_gaussian_hmm_emissions(true_hmm, emissions, true_states, xlim=(0, plot_timesteps))
        plt.show()

    print("log joint prob:    ", true_hmm.log_prob(true_states, emissions))
    print("log marginal prob: ", true_hmm.marginal_log_prob(emissions))


if __name__ == "__main__":
    demo()
