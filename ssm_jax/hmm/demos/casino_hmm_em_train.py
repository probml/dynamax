"""
This demo shows the parameter estimations of HMMs
via Baulm-Welch algorithm on the occasionally dishonest casino example.
Author : Aleyna Kara(@karalleyna)
"""
import jax.numpy as jnp
from jax import random
from jax import vmap
from matplotlib import pyplot as plt
from ssm_jax.hmm.demos.utils import hmm_plot_graphviz
from ssm_jax.hmm.demos.utils import init_random_categorical_hmm
from ssm_jax.hmm.learning import hmm_fit_em
from ssm_jax.hmm.models import CategoricalHMM


def main():
    key = random.PRNGKey(0)
    init_key, sample_key = random.split(key, 2)

    initial_probabilities = jnp.array([1, 1]) / 2

    # state transition matrix
    transition_matrix = jnp.array([[0.95, 0.05], [0.10, 0.90]])

    # observation matrix
    emission_probs = jnp.array([
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],  # fair die
        [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 5 / 10]  # loaded die
    ])

    casino = CategoricalHMM(initial_probabilities, transition_matrix, emission_probs)
    sample_fn = vmap(casino.sample, in_axes=(0, None))
    n, num_timesteps = 4, 5000
    keys = random.split(sample_key, n)
    _, batch_emissions = sample_fn(keys, num_timesteps)

    sizes = emission_probs.shape
    hmm = init_random_categorical_hmm(init_key, sizes)

    hmm_em, losses = hmm_fit_em(hmm, batch_emissions, num_iters=200)
    print(hmm_em.transition_matrix, hmm_em.emission_probs)
    dotfile = hmm_plot_graphviz(hmm_em.transition_matrix, hmm_em.emission_probs)
    dotfile.render("hmm-casino-dot")

    plt.plot(losses[1:])
    plt.title("Expectation Maximization")
    plt.savefig("casino_em_train.png", dpi=300)


if __name__ == "__main__":
    main()
