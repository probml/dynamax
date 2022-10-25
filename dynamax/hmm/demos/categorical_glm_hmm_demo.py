import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from itertools import count
import matplotlib.pyplot as plt

from dynamax.hmm.models import CategoricalRegressionHMM

if __name__ == "__main__":
    keys = map(jr.PRNGKey, count())

    num_states = 2
    num_classes = 3
    feature_dim = 2
    num_timesteps = 5000

    hmm = CategoricalRegressionHMM(num_states, num_classes, feature_dim)
    true_params, _ = hmm.random_initialization(next(keys))
    true_params["transitions"]["transition_matrix"] = jnp.array([[0.95, 0.05], [0.05, 0.95]])

    covariates = jr.normal(next(keys), (num_timesteps, feature_dim))
    states, emissions = hmm.sample(true_params, next(keys), num_timesteps, covariates=covariates)

    # Try fitting it!
    test_hmm = CategoricalRegressionHMM(num_states, num_classes, feature_dim)
    params, param_props = test_hmm.random_initialization(next(keys))
    params, lps = test_hmm.fit_em(params, param_props, emissions, covariates=covariates, num_iters=100)

    # Plot the data and predictions
    # Compute the most likely states
    most_likely_states = test_hmm.most_likely_states(params, emissions, covariates=covariates)
    # flip states (with current random seed, learned states are permuted)
    permuted_states = 1 - most_likely_states

    # Predict the emissions given the true states
    As = params["emissions"]["weights"][most_likely_states]
    bs = params["emissions"]["biases"][most_likely_states]
    predictions = vmap(lambda x, A, b: A @ x + b)(covariates, As, bs)
    predictions = jnp.argmax(predictions, axis=1)

    offsets = 3 * jnp.arange(num_classes)
    plt.imshow(permuted_states[None, :],
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
    plt.xlabel("EM iteration")
    plt.ylabel("log joint probability")

    plt.figure()
    plt.imshow(jnp.vstack((states[None, :], permuted_states[None, :])),
                aspect="auto", interpolation='none', cmap="Greys")
    plt.yticks([0.0, 1.0], ["$z$", "$\hat{z}$"])
    plt.xlabel("time")
    plt.xlim(0, 100)


    print("true log prob: ", hmm.marginal_log_prob(true_params, emissions, covariates=covariates))
    print("test log prob: ", test_hmm.marginal_log_prob(params, emissions, covariates=covariates))

    plt.show()