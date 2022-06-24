"""
This demo does MAP estimation of an HMM using gradient-descent algorithm applied to the log marginal likelihood.
It includes
1. Mini Batch Gradient Descent
2. Full Batch Gradient Descent
3. Stochastic Gradient Descent
Author: Aleyna Kara(@karalleyna
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import random
from jax import vmap
from ssm_jax.hmm.demos.utils import hmm_plot_graphviz
from ssm_jax.hmm.demos.utils import init_random_categorical_hmm
from ssm_jax.hmm.learning import hmm_fit_minibatch_gradient_descent
from ssm_jax.hmm.models import CategoricalHMM


def main():
    key = random.PRNGKey(0)
    init_key, sample_key, train_key = random.split(key, 3)

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
    _, emissions = sample_fn(keys, num_timesteps)

    sizes = emission_probs.shape
    hmm = init_random_categorical_hmm(init_key, sizes)
    num_iters = 400
    learning_rate = 1e-2
    momentum = 0.95
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)

    # Mini Batch Gradient Descent
    batch_size = 2
    hmm_mb, losses_mbgd = hmm_fit_minibatch_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters,
                                                             train_key)

    # Full Batch Gradient Descent
    batch_size = n
    hmm_fb, losses_fbgd = hmm_fit_minibatch_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters,
                                                             train_key)
    # Stochastic Gradient Descent
    batch_size = 1
    hmm_sgd, losses_sgd = hmm_fit_minibatch_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters,
                                                             train_key)

    losses = [losses_sgd, losses_mbgd, losses_fbgd]

    titles = ["Stochastic Gradient Descent", "Mini Batch Gradient Descent", "Full Batch Gradient Descent"]

    dict_figures = {}
    for loss, title in zip(losses, titles):
        filename = title.replace(" ", "_").lower()
        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_title(f"{title}")
        dict_figures[filename] = fig
        plt.savefig(f"{filename}.png", dpi=300)

    dotfile = hmm_plot_graphviz(hmm_sgd.transition_matrix, hmm_sgd.emission_probs)
    dotfile.render("hmm-casino-dot")


if __name__ == "__main__":
    main()
