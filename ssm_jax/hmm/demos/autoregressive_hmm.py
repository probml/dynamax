"""Demo of a simple Gaussian HMM with 2D emissions.
"""
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from ssm_jax.hmm.models.autoregressive_hmm import AutoregressiveHMM

import matplotlib.pyplot as plt
from ssm_jax.plotting import white_to_color_cmap, COLORS, CMAP




def make_hmm(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack(
        [
            jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
            jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
            jnp.zeros((num_states, emission_dim - 2)),
        ]
    )
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)
    return true_hmm


def main(num_timesteps=2000, plot_timesteps=200, test_mode=False):
    true_hmm = make_hmm()
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)

def plot_gaussian_hmm_data(hmm, emissions, states, xlim=None):
    num_timesteps = len(emissions)
    emission_dim = hmm.emission_shape[0]

    # Plot the data superimposed on the generating state sequence
    plt.figure()
    lim = 1.05 * abs(emissions).max()
    plt.imshow(
        states[None, :],
        aspect="auto",
        interpolation="none",
        cmap=CMAP,
        vmin=0,
        vmax=len(COLORS) - 1,
        extent=(0, num_timesteps, -lim, emission_dim * lim),
        alpha=1,
    )

    means = hmm.emission_means[states]
    for d in range(emission_dim):
        plt.plot(emissions[:, d] + lim * d, "-k")
        plt.plot(means[:, d] + lim * d, ":k")

    if xlim is None:
        plt.xlim(0, num_timesteps)
    else:
        plt.xlim(xlim)

    plt.xlabel("time")
    plt.yticks(lim * jnp.arange(emission_dim), ["$y_{}$".format(d + 1) for d in range(emission_dim)])

    plt.title("Simulated data from an HMM")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # Specify parameters of the HMM
    num_states = 5
    emission_dim = 2
    num_lags = 1
    num_timesteps = 1000

    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)

    dynamics_matrices = jnp.tile(0.99 * jnp.eye(emission_dim), [num_states, 1, 1, 1])
    dynamics_biases = 0.01 * jnp.column_stack(
        [
            jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
            jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
            jnp.zeros((num_states, emission_dim - 2)),
        ]
    )
    dynamics_covs = jnp.tile(0.01**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    true_hmm = AutoregressiveHMM(initial_probs,
                                 transition_matrix,
                                 dynamics_matrices,
                                 dynamics_biases,
                                 dynamics_covs)

    true_states, emissions, history = true_hmm.sample(jr.PRNGKey(0), num_timesteps,
                                                      initial_history=jnp.zeros((1, emission_dim)))

    # plt.plot(emissions[:, 0], emissions[:, 1])
    # plt.show()

    from ssm_jax.hmm.learning import hmm_fit_em
    hmm = AutoregressiveHMM.random_initialization(jr.PRNGKey(0), num_states, emission_dim, num_lags)
    hmm, log_probs = hmm_fit_em(hmm, emissions[None, ...], history=history[None, ...])

    plt.plot(log_probs)
    plt.xlabel("EM iteration")
    plt.ylabel("Marginal Log Probability")
    plt.show()
