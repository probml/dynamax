"""Demo of a simple Gaussian HMM with 2D emissions.
"""
import jax.numpy as jnp
import jax.random as jr
import optax

from ssm_jax.hmm.models import GaussianHMM
import ssm_jax.hmm.learning as learning

import matplotlib.pyplot as plt
from ssm_jax.plotting import white_to_color_cmap, COLORS, CMAP


def plot_gaussian_hmm(hmm, emissions, states):
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
    plt.title("Emission Distributions")


def plot_gaussian_hmm_emissions(hmm, emissions, states, xlim=None):
    num_timesteps = len(emissions)
    emission_dim = hmm.emission_shape[0]

    # Plot the data and the smoothed data
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


def plot_hmm_posterior(true_states, posterior):
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
    plt.xlim(0, 200)
    plt.tight_layout()


def demo(num_states=5,
         emission_dim=2,
         num_timesteps=2000,
         num_em_iters=50,
         num_sgd_iters=2000,
         test_mode=False):

    # Specify parameters of the HMM
    initial_probs = jnp.ones(num_states) / num_states
    transition_matrix = 0.95 * jnp.eye(num_states) + 0.05 * jnp.roll(jnp.eye(num_states), 1, axis=1)
    emission_means = jnp.column_stack([
        jnp.cos(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1],
        jnp.sin(jnp.linspace(0, 2 * jnp.pi, num_states+1))[:-1],
        jnp.zeros((num_states, emission_dim - 2))
    ])
    emission_covs = jnp.tile(0.1**2 * jnp.eye(emission_dim), (num_states, 1, 1))

    # Make a true HMM and sample fromt it
    true_hmm = GaussianHMM(initial_probs,
                           transition_matrix,
                           emission_means,
                           emission_covs)
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)

    if not test_mode:
        plot_gaussian_hmm(true_hmm, emissions, true_states)
        plot_gaussian_hmm_emissions(true_hmm, emissions, true_states, xlim=(0, 500))

    print("log joint prob:    ", true_hmm.log_prob(true_states, emissions))
    print("log marginal prob: ", true_hmm.marginal_log_prob(emissions))

    # Fit a GaussianHMM with twice number of true states using EM
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * num_states, emission_dim)
    test_hmm_em, logprobs_em = learning.hmm_fit_em(test_hmm_em, emissions, niter=num_em_iters)

    # Get the posterior
    print("true LL: ", true_hmm.marginal_log_prob(emissions))
    print("EM LL:  ", test_hmm_em.marginal_log_prob(emissions))
    posterior = test_hmm_em.smoother(emissions)
    most_likely_states = test_hmm_em.most_likely_states(emissions)

    if not test_mode:
        plt.figure()
        plt.plot(logprobs_em)
        plt.xlabel("EM iteration")
        plt.ylabel("Marginal log likelihood")

        plot_gaussian_hmm(test_hmm_em, emissions, most_likely_states)
        plot_hmm_posterior(true_states, posterior)

    # Fit a Gaussian HMM with twice number of true states using SGD
    test_hmm_sgd = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    optimizer = optax.adam(learning_rate=1e-2)
    test_hmm_sgd, losses = learning.hmm_fit_sgd(GaussianHMM, test_hmm_sgd, emissions, optimizer,
                                                niter=num_sgd_iters)

    # Get the posterior
    print("true LL: ", true_hmm.marginal_log_prob(emissions))
    print("SGD LL:  ", test_hmm_sgd.marginal_log_prob(emissions))
    posterior = test_hmm_sgd.smoother(emissions)
    most_likely_states = test_hmm_sgd.most_likely_states(emissions)

    # Plot the training curve
    if not test_mode:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        plot_gaussian_hmm(test_hmm_sgd, emissions, most_likely_states)
        plot_hmm_posterior(true_states, posterior)
        plt.show()

if __name__ == "__main__":
    demo()
