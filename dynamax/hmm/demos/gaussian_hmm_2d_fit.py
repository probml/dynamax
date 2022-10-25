"""Demo of fitting a simple Gaussian HMM with 2D emissions.
"""
from cmath import log
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from dynamax.hmm.demos.gaussian_hmm_2d import make_hmm
from dynamax.hmm.demos.gaussian_hmm_2d import plot_gaussian_hmm
from dynamax.hmm.demos.gaussian_hmm_2d import plot_gaussian_hmm_data
from dynamax.hmm.demos.gaussian_hmm_2d import plot_hmm_posterior
from dynamax.hmm.models import GaussianHMM


def main(num_timesteps=2000, plot_timesteps=200, num_em_iters=50, num_sgd_iters=2000, test_mode=False):

    true_hmm, true_params = make_hmm()
    true_states, emissions = true_hmm.sample(true_params, jr.PRNGKey(0), num_timesteps)

    if not test_mode:
        plot_gaussian_hmm(true_hmm, true_params, emissions, true_states, "True HMM")
        plot_gaussian_hmm_data(true_hmm, true_params, emissions, true_states, xlim=(0, plot_timesteps))

    print("log joint prob:    ", true_hmm.log_prob(true_params, true_states, emissions))
    print("log marginal prob: ", true_hmm.marginal_log_prob(true_params, emissions))

    # Fit a GaussianHMM with twice number of true states using EM
    print("Fit with EM")
    test_hmm = GaussianHMM(2 * true_hmm.num_states, true_hmm.emission_dim)
    initial_params, param_props = test_hmm.random_initialization(jr.PRNGKey(1))
    fitted_params_em, logprobs_em = test_hmm.fit_em(initial_params, param_props, emissions, num_iters=num_em_iters)

    # Get the posterior
    print("true LL: ", true_hmm.marginal_log_prob(true_params, emissions))
    print("EM LL:  ", test_hmm.marginal_log_prob(fitted_params_em, emissions))
    posterior = test_hmm.smoother(fitted_params_em, emissions)
    most_likely_states = test_hmm.most_likely_states(fitted_params_em, emissions)

    if not test_mode:
        plt.figure()
        plt.plot(logprobs_em)
        plt.xlabel("EM iteration")
        plt.ylabel("Marginal log likelihood")
        plot_gaussian_hmm(test_hmm, fitted_params_em, emissions, most_likely_states, "EM estimate")
        plot_hmm_posterior(true_states, posterior, plot_timesteps=plot_timesteps)

    # Fit a Gaussian HMM with twice number of true states using SGD
    print("Fit with SGD")
    optimizer = optax.adam(learning_rate=1e-2)
    fitted_params_sgd, logprobs_sgd = test_hmm.fit_sgd(initial_params, param_props, emissions, optimizer=optimizer, num_epochs=num_sgd_iters)

    # Get the posterior
    print("true LL: ", true_hmm.marginal_log_prob(true_params, emissions))
    print("SGD LL:  ", test_hmm.marginal_log_prob(fitted_params_sgd, emissions))
    posterior = test_hmm.smoother(fitted_params_sgd, emissions)
    most_likely_states = test_hmm.most_likely_states(fitted_params_sgd, emissions)

    if not test_mode:
        plt.figure()
        plt.plot(logprobs_sgd)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plot_gaussian_hmm(test_hmm, fitted_params_sgd, emissions, most_likely_states, "SGD estimate")
        plot_hmm_posterior(true_states, posterior, plot_timesteps=plot_timesteps)

    if not test_mode:
        # add plt.show at very end so the demo doesn't wait after each plot when run from command line
        plt.show()


if __name__ == "__main__":
    main()
