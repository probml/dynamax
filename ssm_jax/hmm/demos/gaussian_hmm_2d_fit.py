"""Demo of fitting a simple Gaussian HMM with 2D emissions.
"""
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import ssm_jax.hmm.learning as learning
from ssm_jax.hmm.demos.gaussian_hmm_2d import make_hmm
from ssm_jax.hmm.demos.gaussian_hmm_2d import plot_gaussian_hmm
from ssm_jax.hmm.demos.gaussian_hmm_2d import plot_gaussian_hmm_data
from ssm_jax.hmm.demos.gaussian_hmm_2d import plot_hmm_posterior
from ssm_jax.hmm.models import GaussianHMM
from ssm_jax.plotting import CMAP
from ssm_jax.plotting import COLORS
from ssm_jax.plotting import white_to_color_cmap


def main(num_timesteps=2000, plot_timesteps=200, num_em_iters=50, num_sgd_iters=2000, test_mode=False):

    true_hmm = make_hmm()
    true_states, emissions = true_hmm.sample(jr.PRNGKey(0), num_timesteps)

    if not test_mode:
        plot_gaussian_hmm(true_hmm, emissions, true_states, "True HMM")
        plot_gaussian_hmm_data(true_hmm, emissions, true_states, xlim=(0, plot_timesteps))

    print("log joint prob:    ", true_hmm.log_prob(true_states, emissions))
    print("log marginal prob: ", true_hmm.marginal_log_prob(emissions))

    # Fit a GaussianHMM with twice number of true states using EM
    print("Fit with EM")
    batch_emissions = emissions[None, ...]
    test_hmm_em = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    test_hmm_em, logprobs_em = learning.hmm_fit_em(test_hmm_em, batch_emissions, num_iters=num_em_iters)

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
        plot_gaussian_hmm(test_hmm_em, emissions, most_likely_states, "EM estimate")
        plot_hmm_posterior(true_states, posterior, plot_timesteps)

    # Fit a Gaussian HMM with twice number of true states using SGD
    print("Fit with SGD")
    test_hmm_sgd = GaussianHMM.random_initialization(jr.PRNGKey(1), 2 * true_hmm.num_states, true_hmm.num_obs)
    optimizer = optax.adam(learning_rate=1e-2)
    test_hmm_sgd, losses = learning.hmm_fit_sgd(test_hmm_sgd,
                                                batch_emissions,
                                                optimizer=optimizer,
                                                num_iters=num_sgd_iters)

    # Get the posterior
    print("true LL: ", true_hmm.marginal_log_prob(emissions))
    print("SGD LL:  ", test_hmm_sgd.marginal_log_prob(emissions))
    posterior = test_hmm_sgd.smoother(emissions)
    most_likely_states = test_hmm_sgd.most_likely_states(emissions)

    if not test_mode:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plot_gaussian_hmm(test_hmm_sgd, emissions, most_likely_states, "SGD estimate")
        plot_hmm_posterior(true_states, posterior, plot_timesteps)

    if not test_mode:
        # add plt.show at very end so the demo doesn't wait after each plot when run from command line
        plt.show()


if __name__ == "__main__":
    main()
