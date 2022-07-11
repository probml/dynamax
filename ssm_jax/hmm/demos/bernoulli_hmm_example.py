import jax.numpy as jnp
import jax.numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp
from jax import random
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from ssm_jax.hmm.learning import hmm_fit_em
from ssm_jax.hmm.models.bernoulli_hmm import BernoulliHMM

sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]

colors = sns.xkcd_palette(color_names)


def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    Reference:
    https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/plots.py#L20
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds), 'green': tuple(greens), 'blue': tuple(blues), 'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap


cmap = gradient_cmap(colors)


def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == jnp.int32 and z2.dtype == jnp.int32
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap.at[k1, k2].set(jnp.sum((z1 == k1) & (z2 == k2)))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    # assert jnp.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def plot_transition_matrix(transition_matrix):
    plt.imshow(transition_matrix, vmin=0, vmax=1, cmap="Greys")
    plt.xlabel("next state")
    plt.ylabel("current state")
    plt.colorbar()
    plt.show()


def compare_transition_matrix(true_matrix, test_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    out = axs[0].imshow(true_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[1].imshow(test_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[0].set_title("True Transition Matrix")
    axs[1].set_title("Test Transition Matrix")
    cax = fig.add_axes([
        axs[1].get_position().x1 + 0.07,
        axs[1].get_position().y0,
        0.02,
        axs[1].get_position().y1 - axs[1].get_position().y0,
    ])
    plt.colorbar(out, cax=cax)
    plt.show()


def plot_hmm_data(obs, states):
    lim = 1.01 * abs(obs).max()
    time_bins, obs_dim = obs.shape
    plt.figure(figsize=(8, 3))
    plt.imshow(
        states[None, :],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=len(colors) - 1,
        extent=(0, time_bins, -lim, (obs_dim) * lim),
    )

    for d in range(obs_dim):
        plt.plot(obs[:, d] + lim * d, "-k")

    plt.xlim(0, time_bins)
    plt.xlabel("time")
    plt.yticks(lim * np.arange(obs_dim), ["$x_{}$".format(d + 1) for d in range(obs_dim)])

    plt.title("Simulated data from an HMM")

    plt.tight_layout()


def plot_posterior_states(Ez, states, perm):
    plt.figure(figsize=(25, 5))
    plt.imshow(Ez.T[perm], aspect="auto", interpolation="none", cmap="Greys")
    plt.plot(states, label="True State")
    plt.plot(Ez.T[perm].argmax(axis=0), "--", label="Predicted State")
    plt.xlabel("time")
    plt.ylabel("latent state")
    # plt.legend(bbox_to_anchor=(1,1))
    plt.title("Predicted vs. Ground Truth Latent State")
    # plt.show()


"""# Bernoulli HMM

### Let's create a true model
"""
tfp = tfp.substrates.jax
tfd = tfp.distributions

num_states = 5
num_channels = 10

initial_probabilities = jnp.ones((num_states,)) / (num_states * 1.)
transition_matrix = 0.90 * jnp.eye(num_states) + 0.10 * jnp.ones((num_states, num_states)) / num_states
probs_prior = tfd.Beta(1., 1.)
emission_probabilities = probs_prior.sample(seed=random.PRNGKey(0), sample_shape=(num_states, num_channels))

true_hmm = BernoulliHMM(initial_probabilities, transition_matrix, emission_probabilities)

print(true_hmm.emission_probs)
plot_transition_matrix(true_hmm.transition_matrix)
"""### From the true model, we can sample synthetic data"""

rng = jr.PRNGKey(0)
num_timesteps = 500

states, data = true_hmm.sample(rng, num_timesteps)
"""### Let's view the synthetic data"""

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
axs[0].imshow(data.T, aspect="auto", interpolation="none")
# axs[0].set_ylabel("neuron")
axs[0].set_title("Observations")
axs[1].plot(states)
axs[1].set_title("Latent State")
axs[1].set_xlabel("time")
axs[1].set_ylabel("state")

plt.savefig("bernoulli-hmm-data.pdf")
plt.savefig("bernoulli-hmm-data.png")

emission_probabilities = probs_prior.sample(seed=random.PRNGKey(32), sample_shape=(num_states, num_channels))
transition_matrix = jnp.ones((num_states, num_states)) / (num_states * 1.)

test_hmm = BernoulliHMM(initial_probabilities, transition_matrix, emission_probabilities)

test_hmm, lps = hmm_fit_em(test_hmm, data.reshape((1, 500, 10)), num_iters=20)

# Plot the log probabilities
print("sonuccccc", lps)

print(test_hmm.transition_matrix)

raise "eerrr"

# Compare the transition matrices
compare_transition_matrix(true_hmm.transition_matrix, test_hmm.transition_matrix)
plt.savefig("bernoulli-hmm-transmat-comparison.pdf")

Ez = true_hmm.most_likely_states(data)
perm = find_permutation(states, Ez)
plot_posterior_states(Ez, states, perm)

plt.savefig("bernoulli-hmm-state-est-comparison.pdf")
plt.savefig("bernoulli-hmm-state-est-comparison.png")
plt.show()
"""# Fit Bernoulli Over Multiple Trials"""

rng = jr.PRNGKey(0)
num_timesteps = 500
num_trials = 5

all_states, all_data = true_hmm.sample(rng, num_timesteps, num_samples=num_trials)

# Now we have a batch dimension of size `num_trials`
print(all_states.shape)
print(all_data.shape)

lps, test_hmm, posterior = test_hmm.fit(all_data, method="em", tol=-1)

# plot marginal log probabilities
plt.title("Marginal Log Probability")
plt.ylabel("lp")
plt.xlabel("idx")
plt.plot(lps / data.size)

compare_transition_matrix(true_hmm.transition_matrix, test_hmm.transition_matrix)

# For the first few trials, let's see how good our predicted states are
for trial_idx in range(3):
    print("=" * 5, f"Trial: {trial_idx}", "=" * 5)
    Ez = posterior.expected_states[trial_idx]
    states = all_states[trial_idx]
    perm = find_permutation(states, np.argmax(Ez, axis=-1))
    plot_posterior_states(Ez, states, perm)