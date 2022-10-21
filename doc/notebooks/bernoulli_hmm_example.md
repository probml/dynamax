---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.8.10 64-bit
  language: python
  name: python3
---

+++ {"id": "5mvqmh1jM45B"}

# Bernoulli HMM Example Notebook

Modified from https://github.com/lindermanlab/ssm-jax-refactor/blob/main/notebooks/bernoulli-hmm-example.ipynb

Changes: It uses pyprobml's dynamax library, and finds the permutation between inferrred and true latent state labels in a different way.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: PExYKblDM45N
outputId: 07cf49b3-8287-4fae-f292-93c0560db6ae
---
try:
    import dynamax
except ModuleNotFoundError:
    %pip install git+https://github.com/probml/dynamax.git
    import dynamax
dynamax.__file__
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 4Q6lUKyMzMFF
outputId: 498408e9-b9ad-4aa4-ecd6-8746d3ef8deb
---
try:
    from probml_utils import savefig, latexify, is_latexify_enabled
except ModuleNotFoundError:
    %pip install git+https://github.com/probml/probml-utils.git
    from probml_utils import savefig, latexify, is_latexify_enabled
```

+++ {"id": "yGKxrd0oM45R"}

#### Imports and Plotting Functions 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lSS0U4OTmy1f
outputId: 426e54f0-0f49-48d2-ab30-a89fab04a1fe
---
import jax.random as jr
import jax.numpy as jnp
from jax import vmap

import tensorflow_probability.substrates.jax.distributions as tfd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pprint import pprint

from dynamax.hmm.models import BernoulliHMM
```

```{code-cell} ipython3
:id: WJ7rVLxfmpPS

def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    Reference:
    https://github.com/lindermanlab/ssm/blob/646e1889ec9a7efb37d4153f7034c258745c83a5/ssm/plots.py#L20
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = jnp.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1.0, 1.0))

    cdict = {"red": tuple(reds), "green": tuple(greens), "blue": tuple(blues), "alpha": tuple(alphas)}

    cmap = LinearSegmentedColormap("grad_colormap", cdict, nsteps)
    return cmap
```

```{code-cell} ipython3
:id: yIq0_Y7TnfE1

def find_permutation(z1, z2):
    K1 = z1.max() + 1
    K2 = z2.max() + 1

    perm = []
    for k1 in range(K1):
        indices = jnp.where(z1 == k1)[0]
        counts = jnp.bincount(z2[indices])
        perm.append(jnp.argmax(counts))

    return jnp.array(perm)
```

```{code-cell} ipython3
:id: ro1UeR0b0A6d

sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


def plot_transition_matrix(transition_matrix):
    plt.imshow(transition_matrix, vmin=0, vmax=1, cmap="Greys")
    plt.xlabel("next state")
    plt.ylabel("current state")
    plt.colorbar()
    plt.show()


def compare_transition_matrix(true_matrix, test_matrix):
    # latexify(width_scale_factor=1, fig_height=1.5)
    figsize = (10, 5)
    if is_latexify_enabled():
        figsize = None
    latexify(width_scale_factor=1, fig_height=1.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    out = axs[0].imshow(true_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[1].imshow(test_matrix, vmin=0, vmax=1, cmap="Greys")
    axs[0].set_title("True Transition Matrix")
    axs[1].set_title("Test Transition Matrix")
    cax = fig.add_axes(
        [
            axs[1].get_position().x1 + 0.07,
            axs[1].get_position().y0,
            0.02,
            axs[1].get_position().y1 - axs[1].get_position().y0,
        ]
    )
    plt.colorbar(out, cax=cax)
    plt.show()


def plot_posterior_states(Ez, states, perm):
    # latexify(width_scale_factor=1, fig_height=1.5)
    figsize = (25, 5)
    if is_latexify_enabled():
        figsize = None
    plt.figure(figsize=figsize)
    plt.imshow(Ez.T[perm], aspect="auto", interpolation="none", cmap="Greys")
    plt.plot(states, label="True State", linewidth=1)
    plt.plot(Ez.T[perm].argmax(axis=0), "--", label="Predicted State", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("latent state")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title("Predicted vs. Ground Truth Latent State")
```

+++ {"id": "UJyD2rs2M45V"}

# Bernoulli HMM

+++ {"id": "vGNkra1LM45W"}

### Let's create a true model

```{code-cell} ipython3
:id: KxkZXFobM45X

num_states = 5
num_channels = 10
true_hmm = BernoulliHMM(num_states, num_channels, 
                        emission_prior_concentration0=1.0, emission_prior_concentration1=1.0)
```

```{code-cell} ipython3
params, param_props = true_hmm.random_initialization(jr.PRNGKey(0))
pprint(params)
```

```{code-cell} ipython3
params["initial"]["probs"] = jnp.ones((num_states,)) / (num_states * 1.0)
params["transitions"]["transition_matrix"] = 0.90 * jnp.eye(num_states) + 0.10 * jnp.ones((num_states, num_states)) / num_states
# params["emissions"]["probs"] = probs_prior.sample(seed=jr.PRNGKey(0), sample_shape=(num_states, num_channels))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 308
id: KFw9jPVmpZZt
outputId: e59f75c2-c2f9-459d-9d9c-abb4967f1957
---
plot_transition_matrix(params["transitions"]["transition_matrix"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: PvxApVZjM45b
outputId: a8c89f45-cc1a-4697-a546-2a4c399b3558
---
print("Emission probabilities (num states x num emission_dims (aka channels)):")
print(params["emissions"]["probs"])
```

+++ {"id": "dMyyOCMiM45d"}

### From the true model, we can sample synthetic data

```{code-cell} ipython3
:id: FtB9i2kWM45e

rng = jr.PRNGKey(0)
num_timesteps = 500

states, data = true_hmm.sample(params, rng, num_timesteps)
```

+++ {"id": "HvNTMOuxM45f"}

### Let's view the synthetic data

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 538
id: 958ZS6JVM45f
outputId: bc7d8d05-9257-477a-ab61-4e9049522195
---
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
axs[0].imshow(data.T, aspect="auto", interpolation="none")
axs[0].set_ylabel("channel")
axs[0].set_title("Simulated Data")
axs[1].plot(states)
axs[1].set_title("Latent State")
axs[1].set_xlabel("time")
axs[1].set_ylabel("state")
plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 587
id: 4eIKleZLznTD
outputId: 41b9033d-c185-453c-e0cf-3c3b0d3d5987
---
latexify(width_scale_factor=1, fig_height=2)
figsize = (20, 8)
if is_latexify_enabled():
    figsize = None
fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)
axs[0].imshow(data.T, aspect="auto", interpolation="none")
axs[0].set_ylabel("observable")
axs[0].set_yticks([0, 4, 8])
axs[1].plot(states, linewidth=1)
axs[1].set_xlabel("time")
axs[1].set_ylabel("state")
axs[1].set_yticks([0, 2, 4])
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
if is_latexify_enabled():
    plt.subplots_adjust(hspace=4)
savefig("bernoulli-hmm-data")
plt.show()
```

+++ {"id": "DFVYziNuM45g"}

## Fit HMM using exact EM update

```{code-cell} ipython3
:id: _YSFVQnMM45h

test_hmm = BernoulliHMM(num_states, num_channels)
test_params, test_param_props = test_hmm.random_initialization(jr.PRNGKey(1))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 49
  referenced_widgets: [91c8e1caf9f84f72bb9bc74758a1ae64, 742c8a837eef477db62cfa019c1e7e38,
    778c754a8c8d4219af4ccb982ae4b566, 0e57bbc0bb5f4f7a88ff089e3ba3177f, 5b8cd2b6c8144525ae9f91d761a81d29,
    5e091fb6b95349409e25836129e36ceb, 97a21de0683348c8b3ded6d5b461dfde, d95b15e3e9de46fdb9735382b9e8eb77,
    05427155afed4068af62112dab17d175, 8927208732984d539ccba9186c4b750c, ccac12a17a56467ca10c80296b5b4dbd]
id: 18dUmRQVqPkJ
outputId: 55db5954-6f51-425b-877b-ce3362daab02
---
batch_size = 1
num_iters = 20
test_params, lps = test_hmm.fit_em(test_params, test_param_props, 
                                   data.reshape((batch_size, num_timesteps, num_channels)), 
                                   num_iters=num_iters)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 319
id: pjZLIiUkM45i
outputId: e1d8aee8-aca7-4d99-bfed-6edf9eb35d82
---
# Plot the log probabilities
plt.plot(lps)
plt.xlabel("iteration")
plt.ylabel("log likelihood")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1jKNNGbntrxP
outputId: 3003aa1f-ed4e-49d8-ea02-5bf5301d9ec2
---
test_params["transitions"]["transition_matrix"]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 370
id: D-1ZuywkqST8
outputId: bb726906-b9bd-4ed5-c49b-92a1315ef596
---
# Compare the transition matrices
compare_transition_matrix(params["transitions"]["transition_matrix"], test_params["transitions"]["transition_matrix"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 320
id: JmDSt_YvM45i
outputId: e375a07a-078a-4a29-c6a6-c6994c9ab17a
---
# Posterior distribution
posterior = test_hmm.smoother(test_params, data)
Ez = posterior.smoothed_probs
perm = find_permutation(states, jnp.argmax(Ez, axis=-1))
plot_posterior_states(Ez, states, perm)
```

+++ {"id": "tr-rMyCRM45j"}

# Fit Bernoulli Over Multiple Trials

```{code-cell} ipython3
:id: bWz4ODsqM45k

rng = jr.PRNGKey(0)
num_trials = 5
keys = jr.split(rng, num_trials)
num_timesteps = 500

from functools import partial

all_states, all_data = vmap(partial(true_hmm.sample, params, num_timesteps=num_timesteps))(keys)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1wMbvyTYM45k
outputId: e4d459fb-ed86-43c6-8f2e-995eb330e4c8
---
# Now we have a batch dimension of size `num_trials`
print(all_states.shape)
print(all_data.shape)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 49
  referenced_widgets: [43ca685282574b88807417527c8836d2, 0c732a9554f94b1ea2b0c21b5ea02961,
    17b16abc737b40f3bc88e0fbaaa194ca, 2ad31d8e7be64991826be8915f297dcc, 6dd5f71ff51542de837223c8d6dbbb9c,
    f667913c1e8d4d89bb239271977811f0, e870a10dcd1043ba9ad6cecfa656253a, 57b03daa347c4f638f3e2cbea4e3134d,
    29fc1701894b45308eb647864f2bb50d, 3b5cbad9e01c4c4a826ba0f27cf84b81, 19ec3898adb645698110b92c241d60d9]
id: jo6qXbmjM45l
outputId: 0d51f137-f993-4da0-9cad-525d25263ba0
---
num_iters = 100
test_parms, test_param_props = test_hmm.random_initialization(jr.PRNGKey(2))
test_params, lps = test_hmm.fit_em(test_params, test_param_props, all_data, num_iters=100)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 338
id: mQh9AmrNM45m
outputId: fb1de2a1-d80e-4ce8-9341-4096e0575cbd
---
# plot marginal log probabilities
plt.title("Marginal Log Probability")
plt.ylabel("lp")
plt.xlabel("idx")
plt.plot(jnp.array(lps) / data.size)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 370
id: k3RrRVHVwMq5
outputId: 613d1a45-dea9-4fd7-8d45-37cc5d8c4345
---
# Compare the transition matrices
compare_transition_matrix(params["transitions"]["transition_matrix"], test_params["transitions"]["transition_matrix"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 977
id: BpLXGeNVM45n
outputId: 80eaec9e-a885-43e6-ef9d-a3a104acf058
---
# For the first few trials, let's see how good our predicted states are
for trial_idx in range(3):
    print("=" * 5, f"Trial: {trial_idx}", "=" * 5)
    posterior = test_hmm.smoother(test_params, all_data[trial_idx])
    Ez = posterior.smoothed_probs
    states = all_states[trial_idx]
    perm = find_permutation(states, jnp.argmax(Ez, axis=-1))
    plot_posterior_states(Ez, states, perm)
```
