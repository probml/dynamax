from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from itertools import count
import matplotlib.pyplot as plt

from ssm_jax.lgssm.models import LinearGaussianSSM
from ssm_jax.lgssm.learning import lgssm_fit_em

def main(state_dim=2,
         emission_dim=10,
         num_timesteps=100,
         test_mode=False):
    keys = map(jr.PRNGKey, count())

    true_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    true_states, emissions = true_model.sample(next(keys), num_timesteps)

    if not test_mode:
        # Plot the true states and emissions
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(true_states + jnp.arange(state_dim))
        axs[0].set_ylabel("latent states")
        axs[1].plot(emissions + 3 * jnp.arange(emission_dim))
        axs[1].set_ylabel("data")
        axs[1].set_xlabel("time")

    # Fit an LGSSM with EM
    test_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    posterior_stats, marginal_lls = lgssm_fit_em(test_model, jnp.array([emissions]))

    assert jnp.all(jnp.diff(marginal_lls) > -1e-4)

    if not test_mode:
        plt.figure()
        plt.plot(marginal_lls)
        plt.xlabel("iteration")
        plt.ylabel("marginal log likelihood")

    # Compute predicted emissions
    posterior = test_model.smoother(emissions)
    Ey = posterior.smoothed_means @ test_model.emission_matrix.T + test_model.emission_bias
    Covy = test_model.emission_matrix @ posterior.smoothed_covariances @ test_model.emission_matrix.T \
        + test_model.emission_covariance

    if not test_mode:
        plt.figure(figsize=(10, 4))
        plt.plot(emissions + 3 * jnp.arange(emission_dim))
        plt.plot(Ey + 3 * jnp.arange(emission_dim), '--k')
        for i in range(emission_dim):
            plt.fill_between(jnp.arange(len(emissions)),
                            3 * i + Ey[:, i] - 2 * jnp.sqrt(Covy[:, i, i]),
                            3 * i + Ey[:, i] + 2 * jnp.sqrt(Covy[:, i, i]),
                            color='k', alpha=0.25)
        plt.xlabel("time")
        plt.ylabel("data and predictions")
        plt.show()


if __name__ == "__main__":
    main()