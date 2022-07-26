"""Demo of a simple Gaussian HMM with 2D emissions.
"""
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from ssm_jax.hmm.models import GaussianHMM
from ssm_jax.plotting import CMAP
from ssm_jax.plotting import COLORS
from ssm_jax.plotting import white_to_color_cmap


def plot_gaussian_hmm(hmm, emissions, states, ttl="Emission Distributions"):
    lim = 0.85 * abs(emissions).max()
    XX, YY = jnp.meshgrid(jnp.linspace(-lim, lim, 100), jnp.linspace(-lim, lim, 100))
    grid = jnp.column_stack((XX.ravel(), YY.ravel()))

    plt.figure()
    for k in range(hmm.num_states):
        lls = hmm.emission_distribution(k).log_prob(grid)
        plt.contour(XX, YY, jnp.exp(lls).reshape(XX.shape), cmap=white_to_color_cmap(COLORS[k]))
        plt.plot(emissions[states == k, 0], emissions[states == k, 1], "o", mfc=COLORS[k], mec="none", ms=3, alpha=0.5)

    plt.plot(emissions[:, 0], emissions[:, 1], "-k", lw=1, alpha=0.25)
    plt.xlabel("$y_1$")
    plt.ylabel("$y_2$")
    plt.title(ttl)
    plt.tight_layout()
    return plt.gcf()


def plot_gaussian_hmm_data(hmm, emissions, states, xlim=None):
    num_timesteps = len(emissions)
    emission_dim = hmm.num_obs

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

    means = hmm.emission_means.value[states]
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


def plot_hmm_posterior(true_states, posterior, plot_timesteps=None):
    if plot_timesteps is None:
        plot_timesteps = len(true_states)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].imshow(true_states[None, :],
                  aspect="auto",
                  interpolation="none",
                  cmap=CMAP,
                  vmin=0,
                  vmax=len(COLORS) - 1,
                  alpha=1)
    axs[0].set_yticks([])
    axs[0].set_title("true states")

    axs[1].imshow(posterior.smoothed_probs.T, aspect="auto", interpolation="none", cmap="Greys", vmin=0, vmax=1)
    axs[1].set_ylabel("state")
    axs[1].set_xlabel("time")
    axs[1].set_title("expected states")

    plt.xlim(0, plot_timesteps)
    plt.tight_layout()
    return plt.gcf()


def make_hmm(num_states=5, emission_dim=2):
    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states + 1))[:-1],
        jnp.zeros((num_states, emission_dim - 2)),
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    true_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)
    return true_hmm


def plot_results(true_hmm, emissions, true_states, plot_timesteps):
    dict_figures = {}
    print("log joint prob:    ", true_hmm.log_prob(true_states, emissions))
    print("log marginal prob: ", true_hmm.marginal_log_prob(emissions))
    fig = plot_gaussian_hmm(true_hmm, emissions, true_states, "Generating HMM")
    dict_figures["hmm_gauss_2d_emissions"] = fig
    plot_gaussian_hmm_data(true_hmm, emissions, true_states, xlim=(0, plot_timesteps))
    dict_figures["hmm_gauss_2d_trace"] = fig
    return dict_figures


def main(num_timesteps=2000, plot_timesteps=200, test_mode=False):
    true_hmm = make_hmm()
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)
    if not test_mode:
        dict_figures = plot_results(true_hmm, emissions, true_states, plot_timesteps)
        plt.show()


if __name__ == "__main__":
    main()
