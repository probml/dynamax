"""
This script demonstrates how to use the CategoricalRegressionHMM class.
"""
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt

from dynamax.hidden_markov_model import CategoricalRegressionHMM

if __name__ == "__main__":
    key1, key2, key3, key4 = jr.split(jr.PRNGKey(0), 4)

    num_states = 2
    num_classes = 3
    feature_dim = 10
    num_timesteps = 20000

    hmm = CategoricalRegressionHMM(num_states, num_classes, feature_dim)
    transition_matrix = jnp.array([[0.95, 0.05],
                                   [0.05, 0.95]])
    true_params, _ = hmm.initialize(key=key1, transition_matrix=transition_matrix)

    inputs = jr.normal(key2, (num_timesteps, feature_dim))
    states, emissions = hmm.sample(true_params, key3, num_timesteps, inputs=inputs)

    # Try fitting it!
    test_hmm = CategoricalRegressionHMM(num_states, num_classes, feature_dim)
    params, props = test_hmm.initialize(key=key4)
    params, lps = test_hmm.fit_em(params, props, emissions, inputs=inputs, num_iters=100)

    # Plot the data and predictions
    # Compute the most likely states
    most_likely_states = test_hmm.most_likely_states(params, emissions, inputs=inputs)

    # Predict the emissions given the true states
    As = params["emissions"]["weights"][most_likely_states]
    bs = params["emissions"]["biases"][most_likely_states]
    predictions = vmap(lambda x, A, b: A @ x + b)(inputs, As, bs)
    predictions = jnp.argmax(predictions, axis=1)

    offsets = 3 * jnp.arange(num_classes)
    plt.imshow(most_likely_states[None, :],
               extent=(0, num_timesteps, -3, 3 * num_classes),
               aspect="auto",
               cmap="Greys",
               alpha=0.5)
    plt.plot(emissions)
    plt.plot(predictions, ':k')
    plt.xlim(0, num_timesteps)
    plt.ylim(-0.25, 2.25)
    plt.xlabel("time")
    plt.xlim(0, 100)

    plt.figure()
    plt.plot(lps)
    plt.axhline(hmm.marginal_log_prob(true_params, emissions, inputs), color='k', ls=':')
    plt.xlabel("EM iteration")
    plt.ylabel("log joint probability")

    plt.figure()
    plt.imshow(jnp.vstack((states[None, :], most_likely_states[None, :])),
                aspect="auto", interpolation='none', cmap="Greys")
    plt.yticks([0.0, 1.0], ["$z$", r"$\hat{z}$"])
    plt.xlabel("time")
    plt.xlim(0, 500)


    print("true log prob: ", hmm.marginal_log_prob(true_params, emissions, inputs=inputs))
    print("test log prob: ", test_hmm.marginal_log_prob(params, emissions, inputs=inputs))

    plt.show()
