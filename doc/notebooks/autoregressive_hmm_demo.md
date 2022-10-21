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

+++ {"nbpresent": {"id": "5918355f-c759-41e8-9cc9-64baf78695b3"}}

# Autoregressive (AR) HMM Demo

This notebook demonstrates how to construct and fit a linear autoregressive HMM.
Let $x_t$ denote the observation at time $t$. Let $z_t$ denote the corresponding discrete latent state.

The autoregressive hidden Markov model has the following likelihood,
$$
\begin{align}
x_t \mid x_{t-1}, z_t &\sim
\mathcal{N}\left(A_{z_t} x_{t-1} + b_{z_t}, Q_{z_t} \right).
\end{align}
$$
(Technically, higher-order autoregressive processes are also implemented.) 

```{code-cell} ipython3
---
nbpresent:
  id: 346a61a3-9216-480d-b5b8-39a78782a8c3
---
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from dynamax.hmm.models.autoregressive_hmm import LinearAutoregressiveHMM
from dynamax.plotting import gradient_cmap
from dynamax.utils import random_rotation
from tensorflow_probability.substrates import jax as tfp
```

```{code-cell} ipython3
---
nbpresent:
  id: 346a61a3-9216-480d-b5b8-39a78782a8c3
---
sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink"
]


colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)
```

```{code-cell} ipython3
# Make a transition matrix
num_states = 5
transition_probs = (jnp.arange(num_states)**10).astype(float)
transition_probs /= transition_probs.sum()
transition_matrix = jnp.zeros((num_states, num_states))
for k, p in enumerate(transition_probs[::-1]):
    transition_matrix += jnp.roll(p * jnp.eye(num_states), k, axis=1)
    
plt.imshow(transition_matrix, vmin=0, vmax=1, cmap="Greys")
plt.xlabel("next state")
plt.ylabel("current state")
plt.title("transition matrix")
plt.colorbar()
```

```{code-cell} ipython3
---
nbpresent:
  id: 564edd16-a99d-4329-8e31-98fe1e1cef79
---
# Make observation distributions
emission_dim = 2
num_lags = 1

keys = jr.split(jr.PRNGKey(0), num_states)
angles = jnp.linspace(0, 2 * jnp.pi, num_states, endpoint=False)
theta = jnp.pi / 25 # rotational frequency
weights = jnp.array([0.8 * random_rotation(key, emission_dim, theta=theta) for key in keys])
biases = jnp.column_stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros((num_states, emission_dim - 2))])
covariances = jnp.tile(0.001 * jnp.eye(emission_dim), (num_states, 1, 1))

# Compute the stationary points
stationary_points = jnp.linalg.solve(jnp.eye(emission_dim) - weights, biases)
```

# Plot dynamics functions

```{code-cell} ipython3
if emission_dim == 2:
    lim = 5
    x = jnp.linspace(-lim, lim, 10)
    y = jnp.linspace(-lim, lim, 10)
    X, Y = jnp.meshgrid(x, y)
    xy = jnp.column_stack((X.ravel(), Y.ravel()))

    fig, axs = plt.subplots(1, num_states, figsize=(3 * num_states, 6))
    for k in range(num_states):
        A, b = weights[k], biases[k]
        dxydt_m = xy.dot(A.T) + b - xy
        axs[k].quiver(xy[:, 0], xy[:, 1],
                    dxydt_m[:, 0], dxydt_m[:, 1],
                    color=colors[k % len(colors)])

        
        axs[k].set_xlabel('$x_1$')
        axs[k].set_xticks([])
        if k == 0:
            axs[k].set_ylabel("$x_2$")
        axs[k].set_yticks([])
        axs[k].set_aspect("equal")
                        

    plt.tight_layout()
```

# Sample emissions from the ARHMM

```{code-cell} ipython3
:tags: []

# Make an Autoregressive (AR) HMM
true_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=1)
true_params, _ = true_arhmm.random_initialization(jr.PRNGKey(0))
true_params['initial']['probs'] = jnp.ones(num_states) / num_states
true_params['transitions']['transition_matrix'] = transition_matrix
true_params['emissions']['weights'] = weights
true_params['emissions']['biases'] = biases
true_params['emissions']['covs'] = covariances


time_bins = 10000
true_states, emissions = true_arhmm.sample(true_params, jr.PRNGKey(0), time_bins)

# Compute the lagged emissions (aka features)
features = true_arhmm.compute_covariates(emissions)
```

```{code-cell} ipython3
---
nbpresent:
  id: 0feabc13-812b-4d5e-ac24-f8327ecb4d27
---
fig = plt.figure(figsize=(8, 8))
for k in range(num_states):
    plt.plot(*emissions[true_states==k].T, 'o', color=colors[k],
         alpha=0.75, markersize=3)
    
plt.plot(*emissions[:1000].T, '-k', lw=0.5, alpha=0.2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
# plt.gca().set_aspect("equal")
```

+++ {"nbpresent": {"id": "a58c7a02-2777-4af8-982f-e279bd3bbeb6"}}

Below, we visualize each component of of the observation variable as a time series. The colors correspond to the latent state. The dotted lines represent the stationary point of the the corresponding AR state while the solid lines are the actual observations sampled from the HMM.

```{code-cell} ipython3
---
nbpresent:
  id: 1ec5ac27-2d23-4660-8702-4156f8ffdf39
---
# Plot the emissions and the smoothed emissions
plot_slice = (0, 200)
lim = 1.05 * abs(emissions).max()
plt.figure(figsize=(8, 6))
plt.imshow(true_states[None, :],
           aspect="auto",
           cmap=cmap,
           vmin=0,
           vmax=len(colors)-1,
           extent=(0, time_bins, -lim, (emission_dim)*lim))


Ey = jnp.array(stationary_points)[true_states]
for d in range(emission_dim):
    plt.plot(emissions[:,d] + lim * d, '-k')
    plt.plot(Ey[:,d] + lim * d, ':k')

plt.xlim(plot_slice)
plt.xlabel("time")
plt.yticks(lim * jnp.arange(emission_dim), ["$x_{{{}}}$".format(d+1) for d in range(emission_dim)])

plt.tight_layout()
```

+++ {"nbpresent": {"id": "759699ce-fffa-4667-90af-267122e39f01"}}

# Fit an ARHMM

```{code-cell} ipython3
# Now fit an HMM to the emissions
key1, key2 = jr.split(jr.PRNGKey(0), 2)
test_num_states = num_states

arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=num_lags)
params, props = arhmm.random_initialization(jr.PRNGKey(1))

# Run k-means to find clusters in the data. Use those as initial fixed points.
km = KMeans(num_states).fit(emissions)
params['emissions']['weights'] *= 0
params['emissions']['biases'] = jnp.array(km.cluster_centers_)

fitted_params, lps = arhmm.fit_em(params, props, jnp.expand_dims(emissions, 0), features=jnp.expand_dims(features, 0))
```

```{code-cell} ipython3
# Plot the log likelihoods against the true likelihood, for comparison
true_lp = true_arhmm.marginal_log_prob(true_params, emissions, features=features)
plt.plot(lps, label="EM")
plt.plot(true_lp * jnp.ones(len(lps)), ':k', label="True")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")
plt.show()
```

```{code-cell} ipython3
# # Find a permutation of the states that best matches the true and inferred states
# most_likely_states = posterior.most_likely_states()
# arhmm.permute(find_permutation(true_states[num_lags:], most_likely_states))
# posterior.update()
# most_likely_states = posterior.most_likely_states()
posterior = arhmm.smoother(fitted_params, emissions, features=features)
most_likely_states = arhmm.most_likely_states(fitted_params, emissions, features=features)
```

```{code-cell} ipython3
if emission_dim == 2:
    lim = abs(emissions).max()
    x = jnp.linspace(-lim, lim, 10)
    y = jnp.linspace(-lim, lim, 10)
    X, Y = jnp.meshgrid(x, y)
    xy = jnp.column_stack((X.ravel(), Y.ravel()))

    fig, axs = plt.subplots(2, max(num_states, test_num_states), figsize=(3 * num_states, 6))
    for i, model in enumerate([true_arhmm, arhmm]):
        for j in range(model.num_states):
            A = fitted_params['emissions']['weights'][j]
            b = fitted_params['emissions']['biases'][j]
            dxydt_m = xy.dot(A.T) + b - xy
            axs[i,j].quiver(xy[:, 0], xy[:, 1],
                        dxydt_m[:, 0], dxydt_m[:, 1],
                        color=colors[j % len(colors)])


            axs[i,j].set_xlabel('$x_1$')
            axs[i,j].set_xticks([])
            if j == 0:
                axs[i,j].set_ylabel("$x_2$")
            axs[i,j].set_yticks([])
            axs[i,j].set_aspect("equal")
                        

    plt.tight_layout()
```

```{code-cell} ipython3
# Plot the true and inferred discrete states
plot_slice = (0, 1000)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_states[None,num_lags:], aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(plot_slice)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(posterior.smoothed_probs.T, aspect="auto", interpolation="none", cmap="Greys", vmin=0, vmax=1)
plt.xlim(plot_slice)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
```

```{code-cell} ipython3
# Sample the fitted model
sampled_states, sampled_emissions = arhmm.sample(fitted_params, jr.PRNGKey(0), time_bins)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(8, 8))
for k in range(test_num_states):
    plt.plot(*sampled_emissions[sampled_states==k].T, 'o', color=colors[k % len(colors)],
         alpha=0.75, markersize=3)
    
plt.plot(*sampled_emissions.T, '-k', lw=0.5, alpha=0.2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.gca().set_aspect("equal")
```
