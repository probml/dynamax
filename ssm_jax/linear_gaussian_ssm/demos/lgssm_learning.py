import jax.numpy as jnp
import jax.random as jr
from jax import jit
from itertools import count
import matplotlib.pyplot as plt

from ssm_jax.linear_gaussian_ssm.models import LinearGaussianSSM


def main(state_dim=2, emission_dim=10, num_timesteps=100, test_mode=False, method='MLE'):
    keys = map(jr.PRNGKey, count())

    true_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    true_states, emissions = true_model.sample(next(keys), num_timesteps)

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
    test_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    if method=='SGD':
        neg_marginal_lls = test_model.fit_sgd(jnp.array([emissions]), num_epochs=num_iters*30)
        marginal_lls = -neg_marginal_lls * emissions.size
    elif method in ['MLE', 'MAP']:
        marginal_lls = test_model.fit_em(jnp.array([emissions]), num_iters=num_iters, method=method)

    #assert jnp.all(jnp.diff(marginal_lls) > -1e-4)

    if not test_mode:
        plt.figure()
        plt.xlabel("iteration")
        if method=='SGD':
            plt.plot(marginal_lls[1:], label="estimated")
            plt.plot((true_model.log_prior()+true_model.marginal_log_prob(emissions))\
                * jnp.ones(num_iters*30 - 1), "k:", label="true")
            plt.ylabel("marginal joint probability")
        if method in ['MLE', 'MAP']:
            plt.plot(marginal_lls[1:], label="estimated")
            plt.plot(true_model.marginal_log_prob(emissions) * jnp.ones(num_iters - 1), "k:", label="true")
            plt.ylabel("marginal log likelihood")
        plt.legend()

    # Compute predicted emissions
    posterior = test_model.smoother(emissions)
    smoothed_emissions = posterior.smoothed_means @ test_model.emission_matrix.T + test_model.emission_bias
    smoothed_emissions_cov = (
        test_model.emission_matrix @ posterior.smoothed_covariances @ test_model.emission_matrix.T
        + test_model.emission_covariance
    )
    smoothed_emissions_std = jnp.sqrt(jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))

    if not test_mode:
        spc = 3
        plt.figure(figsize=(10, 4))
        for i in range(emission_dim):
            plt.plot(emissions[:, i] + spc * i, "--k", label="observed" if i == 0 else None)
            ln = plt.plot(smoothed_emissions[:, i] + spc * i, label="smoothed" if i == 0 else None)[0]
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
    print("Learning parameters via MLE:")
    main(method='MLE')
    
    print("Learning parameters via MAP:")
    main(method='MAP')
    
    print("Learning parameters via SGD:")
    main(method='SGD')
