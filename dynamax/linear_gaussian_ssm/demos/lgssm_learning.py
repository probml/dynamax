from itertools import count

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.linear_gaussian_ssm.models.linear_gaussian_ssm_conjugate import LinearGaussianConjugateSSM


def main(state_dim=2, emission_dim=10, num_timesteps=100, test_mode=False, method='EM'):
    keys = map(jr.PRNGKey, count())

    true_model = LinearGaussianConjugateSSM(state_dim, emission_dim)
    true_params, param_props = true_model.random_initialization(next(keys))
    true_states, emissions = true_model.sample(true_params, next(keys), num_timesteps)

    if not test_mode:
        # Plot the true states and emissions
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(true_states + jnp.arange(state_dim))
        axs[0].set_ylabel("latent states")
        axs[0].set_xlim(0, num_timesteps - 1)
        axs[1].plot(emissions + 3 * jnp.arange(emission_dim))
        axs[1].set_ylabel("data")
        axs[1].set_xlabel("time")
        axs[1].set_xlim(0, num_timesteps - 1)

    # Fit an LGSSM with EM
    num_iters = 100
    test_model = LinearGaussianConjugateSSM(state_dim, emission_dim)
    test_params, param_props = test_model.random_initialization(next(keys))

    if method == 'SGD':
        test_params, neg_marginal_lls = test_model.fit_sgd(test_params, param_props, emissions, num_epochs=num_iters * 30)
        marginal_lls = -neg_marginal_lls * emissions.size

    elif method == 'EM':
        test_params, marginal_lls = test_model.fit_em(test_params, param_props, emissions, num_iters=num_iters)
        assert jnp.all(jnp.diff(marginal_lls) > -1e-4)

    else:
        raise Exception("Invalid method {}".format(method))

    if not test_mode:
        plt.figure()
        plt.xlabel("iteration")
        if method == 'SGD':
            plt.plot(marginal_lls, label="estimated")
            plt.plot((true_model.log_prior() + true_model.marginal_log_prob(true_params, emissions)) *
                     jnp.ones(num_iters * 30 - 1),
                     "k:",
                     label="true")
            plt.ylabel("marginal joint probability")
        if method == 'EM':
            plt.plot(marginal_lls, label="estimated")
            plt.plot(true_model.marginal_log_prob(true_params, emissions) * jnp.ones(num_iters - 1),
                     "k:",
                     label="true")
            plt.ylabel("marginal log likelihood")
        plt.legend()

    # Compute predicted emissions
    posterior = test_model.smoother(test_params, emissions)
    smoothed_emissions = posterior.smoothed_means @ test_params['emissions']['weights'].T \
        + test_params['emissions']['bias']
    smoothed_emissions_cov = (test_params['emissions']['weights']
                              @ posterior.smoothed_covariances
                              @ test_params['emissions']['weights'].T
                              + test_params['emissions']['cov'])
    smoothed_emissions_std = jnp.sqrt(
        jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))

    if not test_mode:
        spc = 3
        plt.figure(figsize=(10, 4))
        for i in range(emission_dim):
            plt.plot(emissions[:, i] + spc * i, "--k", label="observed" if i == 0 else None)
            ln = plt.plot(smoothed_emissions[:, i] + spc * i,
                          label="smoothed" if i == 0 else None)[0]
            plt.fill_between(
                jnp.arange(num_timesteps),
                spc * i + smoothed_emissions[:, i] - 2 * jnp.sqrt(smoothed_emissions_std[i]),
                spc * i + smoothed_emissions[:, i] + 2 * jnp.sqrt(smoothed_emissions_std[i]),
                color=ln.get_color(),
                alpha=0.25,
            )
        plt.xlabel("time")
        plt.xlim(0, num_timesteps - 1)
        plt.ylabel("true and predicted emissions")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print("Learning parameters via EM:")
    main(method='EM', test_mode=False)

    # print("Learning parameters via SGD:")
    # main(method='SGD')
