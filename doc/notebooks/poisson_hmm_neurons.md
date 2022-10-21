---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "Y9R_8oOH4xwa"}

# Poisson HMM Example Notebook

Based on https://github.com/lindermanlab/dynamax/blob/main/notebooks/poisson-hmm-example.ipynb

```{code-cell} ipython3
:id: BzdqAUq74xwm

try:
    import dynamax
except ModuleNotFoundError:
    %pip install -qq git+https://github.com/probml/dynamax.git
    import dynamax
```

+++ {"id": "nJYRbHej4xwp"}

#### Imports and Plotting Functions 

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: tkTeSCes5uLq
outputId: e8fd84b4-6f14-47d3-848e-596d2cc68de5
---
import warnings
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax import jit

import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from dynamax.hmm.models import PoissonHMM
```

```{code-cell} ipython3
:id: Uu7KAUow8PNI

def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == jnp.int32 and z2.dtype == jnp.int32
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = jnp.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap = overlap.at[k1, k2].set(jnp.sum((z1 == k1) & (z2 == k2)))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    # assert jnp.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = jnp.array(list(set(jnp.arange(K2)) - set(perm)))
        perm = jnp.concatenate((perm, unused))

    return perm
```

```{code-cell} ipython3
:id: xUSaiJy95r1Y

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
:id: tmOo1YR64xwr

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
    fig, axs = plt.subplots(1, 2)
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
    plt.yticks(lim * jnp.arange(obs_dim), ["$x_{}$".format(d + 1) for d in range(obs_dim)])

    plt.title("Simulated data from an HMM")

    plt.tight_layout()


def plot_posterior_states(Ez, states, perm):
    plt.figure(figsize=(20, 2))
    plt.imshow(Ez.T[perm], aspect="auto", interpolation="none", cmap="Greys")
    plt.plot(states, label="True State")
    plt.plot(Ez.T[perm].argmax(axis=0), "--", label="Predicted State")
    plt.xlabel("time")
    plt.ylabel("latent state")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title("Predicted vs. Ground Truth Latent State")
    plt.show()
```

+++ {"id": "U2hlROTr4xwv"}

# 2. Poisson HMM

+++ {"id": "BUUR2juP4xwx"}

### As before, let's create a true model

```{code-cell} ipython3
:id: 7_bgQbVD4xwy

num_states = 5
num_neurons = 10
hmm = PoissonHMM(num_states, num_neurons)

true_params, param_props = hmm.random_initialization(jr.PRNGKey(0))
true_params["initial"]["probs"] = jnp.ones((num_states,)) / (num_states * 1.0)
true_params["transitions"]["transition_matrix"] = 0.90 * jnp.eye(num_states) + 0.10 * jnp.ones((num_states, num_states)) / num_states

means_prior = tfd.Gamma(3.0, 1.0)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    true_params["emissions"]["rates"] = means_prior.sample(seed=jr.PRNGKey(0), sample_shape=(num_states, num_neurons))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 308
id: Y4bc-Gy67T8b
outputId: f940de42-a7dd-4efa-ad4d-8820d10b42f9
---
plot_transition_matrix(true_params["transitions"]["transition_matrix"])
```

+++ {"id": "xuPqfHa84xw2"}

### Acessing model parameters
The HMM was initialized with a transition matrix, but what about the emission rates? Unless they are explicitly given to the constructor, they are initialized randomly using the specified seed.  We can access them via model properties.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lw-no7DZ4xw4
outputId: fa692b6b-78cd-40b9-f261-2df06fae625e
---
print("Emission probabilities (num states x num emission_dims (aka channels)):")
print(true_params["emissions"]["rates"])
```

+++ {"id": "tO4Oo_BS4xw6"}

### From the true model, we can sample synthetic data

```{code-cell} ipython3
:id: L3_GVE924xw7

rng = jr.PRNGKey(0)
num_timesteps = 2000

# There's an annoying warning/bug when sampling from Poisson TFP distributions
# currently. We can suppress the warning here for convenience
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "Explicitly requested dtype")
    states, emissions = hmm.sample(true_params, rng, num_timesteps)
```

+++ {"id": "5DXq5tGX4xw-"}

### Let's view the synthetic data

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 468
id: RIyEoXU74xw-
outputId: 52e7e663-3a55-48e9-b77b-7af2180b5383
---
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
axs[0].imshow(emissions.T, aspect="auto", interpolation="none")
axs[0].set_ylabel("neuron")
axs[0].set_title("Neuron Spiking Activity")
axs[1].plot(states)
axs[1].set_title("Latent State")
axs[1].set_xlabel("time")
axs[1].set_ylabel("state")
plt.show()
```

+++ {"id": "w_71DiQt4xxA"}

## Fit HMM using exact EM update

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 49
  referenced_widgets: [c6d73839393145d4ae9a7b2f236a523a, b3c5cc40c95a44228feaa12df0811c59,
    c857e055b8d048b593f980a3f3851d17, d24bb1a009f947a49dec577994284d1b, a767a52d79524ce5846a12b625161380,
    4ecf70f01e7e4f91ba2423b75e1d77c5, ea445a9a321347f58a6193e93977fa59, ddf076b654564f50827fbd58bcf691bb,
    c38d6fff7e084bb6b04d6c1967836921, 61b550285fcc49338653beba247a75ad, 6ccb20a6d4304d3cb6cd519a0fb74308]
id: HbJHCDmz4xxB
outputId: ef289a70-4966-41ed-84cb-5b8a4db2f23d
---
test_params, param_props = hmm.random_initialization(jr.PRNGKey(1234))
test_hmm, lps = hmm.fit_em(test_params, param_props, emissions[None, ...])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 319
id: W0Po_E-q4xxB
outputId: d4fdb461-a0de-4b1f-e489-277bee6f0a43
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
  height: 234
id: Jt-4CieY735F
outputId: 913442a8-1800-45d1-a84f-6180e1a60474
---
# Compare the transition matrices
compare_transition_matrix(true_params["transitions"]["transition_matrix"], 
                          test_params["transitions"]["transition_matrix"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 162
id: MIPmFDIP4xxC
outputId: 930103fc-f434-4cd1-b727-f7429a9f0e2e
---
# Posterior distribution
posterior = hmm.smoother(test_params, emissions)
Ez = posterior.smoothed_probs
perm = find_permutation(states, jnp.argmax(Ez, axis=-1))
plot_posterior_states(Ez[:100], states[:100], perm)
```

+++ {"id": "_KO-_3r14xxE"}

## Fit Poisson HMM over multiple trials

```{code-cell} ipython3
:id: mx4USrcH4xxG

key = jr.PRNGKey(0)
num_trials = 5
num_timesteps = 500

# once again, we suppress the warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "Explicitly requested dtype")
    all_states, all_emissions = vmap(partial(hmm.sample, true_params, num_timesteps=num_timesteps))(
        jr.split(key, num_trials))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VDpA2vEP4xxG
outputId: e52ffad0-3367-4332-f742-a4020a56448c
---
# Now we have a batch dimension of size `num_trials`
print(all_states.shape)
print(all_emissions.shape)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 49
  referenced_widgets: [5c72647d913944cf87dcdd5d4c96a303, f71af488029b49bb9eccdb90f5737cd9,
    3268e2e08c8c4c2abcf9dbc60b6eda49, de144b24143e4871a82c3abcf7f60db4, 68565668b8e0482ebe0a11ffd919303f,
    e09b6011b8eb4b2db49ff4be63e4e7f6, 7779d259aa5e4d9d889cab4727d53333, 467922ada9974831b414297b4589a14c,
    2f6708f554eb4036bb1c76f20b27bae0, da7ff836a6714dfb931a578143e09c4e, 03f3d7c7fb3143faa19e8c48fbdb4c7b]
id: kQiuBr-a4xxH
outputId: 3c507738-4ef7-47ca-d506-1d251850a6b8
---
test_params, param_props = hmm.random_initialization(jr.PRNGKey(2))
test_params, lps = hmm.fit_em(test_params, param_props, all_emissions, num_iters=100)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 538
id: A80dqlpl4xxI
outputId: d0ec9dbc-415f-4090-92f6-d5b04a914dd0
---
# plot marginal log probabilities
plt.title("Marginal Log Probability")
plt.ylabel("lp")
plt.xlabel("idx")
plt.plot(jnp.array(lps) / all_emissions.size)

compare_transition_matrix(true_params["transitions"]["transition_matrix"], 
                          test_params["transitions"]["transition_matrix"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 505
id: xyPxpYYc4xxJ
outputId: e9e5fcdf-9b61-4425-882a-f5ba055ac4d6
---
for trial_idx in range(3):
    print("=" * 5, f"Trial: {trial_idx}", "=" * 5)
    posterior = hmm.smoother(test_params, all_emissions[trial_idx])
    Ez = posterior.smoothed_probs
    states = all_states[trial_idx]
    perm = find_permutation(states, jnp.argmax(Ez, axis=-1))
    plot_posterior_states(Ez, states, perm)
```
