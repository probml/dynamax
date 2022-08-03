import pytest

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from itertools import count
import matplotlib.pyplot as plt

from ssm_jax.hmm.models import LinearRegressionHMM

# def test_create_linreg_hmm(key=jr.PRNGKey(0), num_states=4, emission_dim=2, feature_dim=3):

if __name__ == "__main__":
    keys = map(jr.PRNGKey, count())

    num_states = 2
    emission_dim = 3
    feature_dim = 2
    num_timesteps = 1000

    hmm = LinearRegressionHMM.random_initialization(next(keys), num_states, emission_dim, feature_dim)
    hmm.emission_covariance_matrices.value *= 0.1

    # features = jr.normal(next(keys), (num_timesteps, feature_dim))
    features = jnp.column_stack([jnp.cos(2 * jnp.pi * jnp.arange(num_timesteps) / 10),
                                 jnp.sin(2 * jnp.pi * jnp.arange(num_timesteps) / 10)])
    states, emissions = hmm.sample(next(keys), num_timesteps, features=features)

    # Try fitting it!
    test_hmm = LinearRegressionHMM.random_initialization(next(keys), num_states, emission_dim, feature_dim)
    lps = test_hmm.fit_em(jnp.expand_dims(emissions, 0), batch_features=jnp.expand_dims(features, 0))

    # Plot the data and predictions
    # Compute the most likely states
    most_likely_states = test_hmm.most_likely_states(emissions, features=features)

    # Predict the emissions given the true states
    As = test_hmm.emission_matrices.value[most_likely_states]
    bs = test_hmm.emission_biases.value[most_likely_states]
    predictions = vmap(lambda x, A, b: A @ x + b)(features, As, bs)

    offsets = 3 * jnp.arange(emission_dim)
    plt.imshow(most_likely_states[None, :],
               extent=(0, num_timesteps, -3, 3 * emission_dim),
               aspect="auto",
               cmap="Greys",
               alpha=0.5)
    plt.plot(emissions + offsets)
    plt.plot(predictions + offsets, ':k')
    plt.xlim(0, num_timesteps)
    plt.ylim(-3, 3 * emission_dim)
    plt.xlabel("time")
    plt.xlim(0, 100)

    plt.figure()
    plt.plot(lps)
    plt.xlabel("EM iteration")
    plt.ylabel("log joint probability")

    print("true log prob: ", hmm.marginal_log_prob(emissions, features=features))
    print("test log prob: ", test_hmm.marginal_log_prob(emissions, features=features))

    plt.show()