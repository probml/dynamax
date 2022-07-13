"""
Using a Hidden Markov Model with Poisson Emissions to Understand Earthquakes

Based on
https://github.com/hmmlearn/hmmlearn/blob/main/examples/plot_poisson_hmm.py
https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_poisson_hmm.html

"""
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from ssm_jax.hmm.learning import hmm_fit_em
from ssm_jax.hmm.models.poisson_hmm import PoissonHMM

# earthquake data from http://earthquake.usgs.gov/
earthquakes = jnp.array([
    13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22,
    19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34,
    10, 15, 22, 18, 15, 20, 15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15, 18, 14, 10, 15, 8, 15, 6,
    11, 8, 7, 18, 16, 13, 12, 13, 20, 15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11
])

# Plot the sampled data
fig, ax = plt.subplots()
ax.plot(earthquakes, ".-", ms=6, mfc="orange", alpha=0.7)
ax.set_xticks(range(0, earthquakes.size, 10))
ax.set_xticklabels(range(1906, 2007, 10))
ax.set_xlabel('Year')
ax.set_ylabel('Count')
fig.show()

emission_dim = 1

# %%
# Now, fit a Poisson Hidden Markov Model to the data.
scores = list()
models = list()

for num_states in range(1, 5):
    for idx in range(10):  # ten different random starting states

        key = jr.PRNGKey(idx)

        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_rates = jr.uniform(key3, (num_states, emission_dim), minval=10., maxval=35.)
        emission_log_rates = jnp.log(emission_rates)

        model = PoissonHMM(initial_probs, transition_matrix, emission_log_rates)
        model, *_ = hmm_fit_em(model, earthquakes[None, ..., None], num_iters=20)
        models.append(model)
        scores.append(model.marginal_log_prob(earthquakes[:, None]))
        print(f'Score: {scores[-1]}')

# get the best model
model = models[jnp.argmax(jnp.array(scores))]
print(f'The best model had a score of {max(scores)} and '
      f'{model.num_states} components')

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = model.most_likely_states(earthquakes[:, None])
# %%
# Let's plot the waiting times from our most likely series of states of
# earthquake activity with the earthquake data. As we can see, the
# model with the maximum likelihood had different states which may reflect
# times of varying earthquake danger.

# plot model states over time
fig, ax = plt.subplots()
ax.plot(model.emission_rates[states], ".-", ms=6, mfc="orange")
ax.plot(earthquakes)
ax.set_title('States compared to generated')
ax.set_xlabel('State')
plt.savefig("earthquake_states")

# %%
# Fortunately, 2006 ended with a period of relative tectonic stability, and,
# if we look at our transition matrix, we can see that the off-diagonal terms
# are small, meaning that the state transitions are rare and it's unlikely that
# there will be high earthquake danger in the near future.

fig, ax = plt.subplots()
ax.imshow(model.transition_matrix, aspect='auto', cmap='spring')
ax.set_title('Transition Matrix')
ax.set_xlabel('State To')
ax.set_ylabel('State From')
plt.savefig("earthquake_transition_matrix")
