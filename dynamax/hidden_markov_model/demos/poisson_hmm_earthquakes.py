"""
Using a Hidden Markov Model with Poisson Emissions to Understand Earthquakes

Based on
https://github.com/hmmlearn/hmmlearn/blob/main/examples/plot_poisson_hmm.py
https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_poisson_hmm.html

"""
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from dynamax.hidden_markov_model.models.poisson_hmm import PoissonHMM

# earthquake data from http://earthquake.usgs.gov/
EARTHQUAKES = jnp.array(
    [13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14, 8, 11, 14,
    23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27,
    35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20, 15, 22, 19, 16, 30,
    27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15, 18, 14, 10,
    15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20, 15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11])



def main(test_mode=False, num_iters=20, num_repeats=10, min_states=2, max_states=4):
    """
    Fit a Poisson Hidden Markov Model to earthquake data.
    """
    emission_dim = 1
    emissions = EARTHQUAKES.reshape(-1, emission_dim)

    # Now, fit a Poisson Hidden Markov Model to the data.
    scores = list()
    models = list()
    model_params = list()

    for num_states in range(min_states, max_states+1):
        for idx in range(num_repeats):  # ten different random starting states
            key = jr.PRNGKey(idx)
            key1, key2 = jr.split(key, 2)

            model = PoissonHMM(num_states, emission_dim)
            params, param_props = model.initialize(key1)
            params["emissions"]["rates"] = jr.uniform(key2, (num_states, emission_dim), minval=10.0, maxval=35.0)

            params, losses = model.fit_em(params, param_props, emissions[None, ...], num_iters=num_iters)
            models.append(model)
            model_params.append(params)
            scores.append(model.marginal_log_prob(params, emissions))
            print(f"Score: {scores[-1]}")

    # get the best model
    model = models[jnp.argmax(jnp.array(scores))]
    params = model_params[jnp.argmax(jnp.array(scores))]
    print(f"The best model had a score of {max(scores)} and "
            f"{model.num_states} components")

    # use the Viterbi algorithm to predict the most likely sequence of states
    # given the model
    states = model.most_likely_states(params, emissions)

    if not test_mode:
        # Let's plot the rates from our most likely series of states of
        # earthquake activity with the earthquake data. As we can see, the
        # model with the maximum likelihood had different states which may reflect
        # times of varying earthquake danger.

        # plot model states over time
        fig, ax = plt.subplots()
        ax.plot(params["emissions"]["rates"][states], ".-", ms=6, mfc="orange")
        ax.plot(emissions.ravel())
        ax.set_title("States compared to generated")
        ax.set_xlabel("State")

        # Fortunately, 2006 ended with a period of relative tectonic stability, and,
        # if we look at our transition matrix, we can see that the off-diagonal terms
        # are small, meaning that the state transitions are rare and it's unlikely that
        # there will be high earthquake danger in the near future.
        fig, ax = plt.subplots()
        ax.imshow(params["transitions"]["transition_matrix"], aspect="auto", cmap="spring")
        ax.set_title("Transition Matrix")
        ax.set_xlabel("State To")
        ax.set_ylabel("State From")

        plt.show()

# Run the demo
if __name__ == "__main__":
    main()
