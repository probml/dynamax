import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from itertools import count
import matplotlib.pyplot as plt

from ssm_jax.hmm.models import CategoricalRegressionHMM

if __name__ == "__main__":
    keys = map(jr.PRNGKey, count())

    num_states = 2
    emission_dim = 3
    feature_dim = 2
    num_timesteps = 5000

    hmm = CategoricalRegressionHMM.random_initialization(next(keys), num_states, emission_dim, feature_dim)
    tmat = jnp.array([[0.95, 0.05], [0.05, 0.95]])
    hmm.transition_matrix.value = tmat
    features = jr.normal(next(keys), (num_timesteps, feature_dim))
    # features = jnp.column_stack([jnp.cos(2 * jnp.pi * jnp.arange(num_timesteps) / 10),
    #                              jnp.sin(2 * jnp.pi * jnp.arange(num_timesteps) / 10)])
    states, emissions = hmm.sample(next(keys), num_timesteps, features=features)

    # Try fitting it!
    test_hmm = CategoricalRegressionHMM.random_initialization(next(keys), num_states, emission_dim, feature_dim)
    lps = test_hmm.fit_em(jnp.expand_dims(emissions, 0), num_iters=100, features=jnp.expand_dims(features, 0))

    # Plot the data and predictions
    # Compute the most likely states
    most_likely_states = test_hmm.most_likely_states(emissions, features=features)
    # flip states (with current random seed, learned states are permuted)
    permuted_states = 1 - most_likely_states 

    # Predict the emissions given the true states
    As = test_hmm.emission_matrices.value[most_likely_states]
    bs = test_hmm.emission_biases.value[most_likely_states]
    predictions = vmap(lambda x, A, b: A @ x + b)(features, As, bs)
    predictions = jnp.argmax(predictions, axis=1)
    
    offsets = 3 * jnp.arange(emission_dim)
    plt.imshow(permuted_states[None, :],
               extent=(0, num_timesteps, -3, 3 * emission_dim),
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


    print("true log prob: ", hmm.marginal_log_prob(emissions, features=features))
    print("test log prob: ", test_hmm.marginal_log_prob(emissions, features=features))

    plt.show()