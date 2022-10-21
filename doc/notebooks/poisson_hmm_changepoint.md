---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.9.6 ('dynamax')
  language: python
  name: python3
---

+++ {"id": "CerscXVN31Wi"}

# HMM with Poisson observations for detecting changepoints in the rate of a signal

Based on https://www.tensorflow.org/probability/examples/Multiple_changepoint_detection_and_Bayesian_model_selection

```{code-cell} ipython3
:id: n6VSJM1gzlM0

import logging

logging.getLogger("absl").setLevel(logging.CRITICAL)
```

```{code-cell} ipython3
:id: ngzILtRd5iEx

from jax.config import config
config.update("jax_debug_nans", True)
```

```{code-cell} ipython3
:id: qf_3RpxOAKX2

try:
    import dynamax
except ModuleNotFoundError:
    %pip install -qq git+https://github.com/probml/dynamax.git
    import dynamax
```

```{code-cell} ipython3
:id: eceu-i1bPOHZ

import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap, value_and_grad

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import optax
from itertools import count
from functools import partial
from matplotlib import pylab as plt

from dynamax.hmm.models import PoissonHMM, StandardHMM
```

+++ {"id": "U5D7a3SA4lBP"}

## Data

+++ {"id": "jFyTZElr4p-K"}

The synthetic data corresponds to a single time series of counts, where the rate of the underlying generative process changes at certain points in time.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 283
id: 8TQh8kvg3Pnn
outputId: 33e5662a-6334-4b11-d7a7-659a19e276cc
---
true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]
keys = map(jr.PRNGKey, count())

emissions = jnp.concatenate(
    [
        jr.poisson(key, rate, (num_steps,))
        for (key, rate, num_steps) in zip(keys, true_rates, true_durations)
    ]
).astype(jnp.float32)

# PoissonHMM requires are least 1D emissions
emissions = emissions[:, None]

plt.plot(emissions)
plt.xlabel("time")
plt.ylabel("count")
```

# Make a Poisson HMM with a log normal prior

We could use a gamma prior, but for the purposes of illustration, we'll make a Poisson HMM with a nonconjugate, log normal prior instead.

```{code-cell} ipython3
class NonconjugatePoissonHMM(PoissonHMM):
    """A Poisson HMM with a nonconjugate prior.    
    """
    def __init__(self, num_states, emission_dim, 
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_loc=0.0,
                 emission_prior_scale=1.0):
        StandardHMM.__init__(self,
            num_states, 
            initial_probs_concentration=initial_probs_concentration,
            transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_loc = emission_prior_loc
        self.emission_prior_scale = emission_prior_scale
        
    def log_prior(self, params):
        return tfd.LogNormal(self.emission_prior_loc, self.emission_prior_scale).log_prob(
            params["emissions"]["rates"]
        ).sum()
        
    # Default to the standard E and M steps rather than the conjugate updates
    # for the PoissonHMM with a gamma prior.
    def e_step(self, params, batch_emissions):
        return StandardHMM.e_step(self, params, batch_emissions)
    
    def m_step(self, params, param_props, batch_emissions, batch_posteriors, **batch_covariates):
        return StandardHMM.m_step(self, params, param_props, batch_emissions, batch_posteriors, **batch_covariates)
```

+++ {"id": "ZFR_nwenIqfZ"}

## Model with fixed $K$

+++ {"id": "wIfy7113It50"}

To model the changing Poisson rate, we use an HMM. We initially assume the number of states is known to be $K=4$. Later we will try comparing HMMs with different $K$.

We fix the initial state distribution to be uniform, and fix the transition matrix to be the following, where we set $p=0.05$:

$$ \begin{align*} z_1 &\sim \text{Categorical}\left(\left\{\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\right\}\right)\\ z_t | z_{t-1} &\sim \text{Categorical}\left(\left\{\begin{array}{cc}p & \text{if } z_t = z_{t-1} \\ \frac{1-p}{4-1} & \text{otherwise}\end{array}\right\}\right) \end{align*}$$

```{code-cell} ipython3
:id: N-qzSZ5Pf_ff

def build_latent_state(num_states, max_num_states, daily_change_prob):
    # Give probability 0 to states outside of the current model.
    def prob(s):
        return jnp.where(s < num_states + 1, 1 / num_states, 0.0)

    states = jnp.arange(1, max_num_states + 1)
    initial_state_probs = vmap(prob)(states)

    # Build a transition matrix that transitions only within the current
    # `num_states` states.
    def transition_prob(i, s):
        return jnp.where(
            (s <= num_states) & (i <= num_states) & (1 < num_states),
            jnp.where(s == i, 1 - daily_change_prob, daily_change_prob / (num_states - 1)),
            jnp.where(s == i, 1, 0),
        )

    transition_probs = vmap(transition_prob, in_axes=(None, 0))(states, states)

    return initial_state_probs, transition_probs
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 9kj3K0ZNMgMt
outputId: 6546a224-b78d-4ac3-94f1-ad3385d13a70
---
num_states = 4
daily_change_prob = 0.05

initial_state_probs, transition_probs = build_latent_state(num_states, num_states, daily_change_prob)
print("Initial state probs:\n{}".format(initial_state_probs))
print("Transition matrix:\n{}".format(transition_probs))
```

+++ {"id": "pFN6ke-9I8W9"}

Now we create an HMM where the observation distribution is a Poisson with learnable parameters. We specify the parameters in log space and initialize them to random values around the log of the overall mean count (to set the scal

+++ {"id": "wXJ7uij9JFLF"}

## Model fitting using Gradient Descent

+++ {"id": "yPZ-z0caJScE"}

We compute a MAP estimate of the Poisson rates $\lambda$ using batch gradient descent, using the Adam optimizer applied to the log likelihood (from the HMM) plus the log prior for $p(\lambda)$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: gSjyTtkDrOHu
outputId: 4df6d98a-54b6-4fa6-a54f-0d075f7e7b3f
---
# Define variable to represent the unknown log rates.
hmm = NonconjugatePoissonHMM(num_states, 
                             emission_dim=1, 
                             emission_prior_loc=3.0, 
                             emission_prior_scale=1.0)
params, param_props = hmm.random_initialization(next(keys))

# Set and freeze the initial distribution and transition matrix
params["initial"]["probs"] = initial_state_probs
params["transitions"]["transition_matrix"] = transition_probs
params["emissions"]["rates"] = tfd.LogNormal(jnp.log(emissions.mean()), 1).sample(
    seed=next(keys), sample_shape=(num_states, 1))
param_props["initial"]["probs"].trainable = False
param_props["transitions"]["transition_matrix"].trainable = False

# Fit the model with SGD
optimizer = optax.adam(1e-1)
num_epochs = 1000
params, losses = hmm.fit_sgd(params,
                     param_props,
                     emissions[None, ...],
                     optimizer=optimizer,
                     num_epochs=num_epochs)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 266
id: 1qEQfk0WJqGW
outputId: 0af67f4d-fc9f-4095-ca03-ee25c61c79e8
---
plt.plot(losses)
plt.ylabel("Negative log marginal likelihood")
plt.xlabel("iteration")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: TljgKrxri_Px
outputId: fa436635-8bd3-46aa-ff5a-c26461e4c9f9
---
print("Inferred rates: {}".format(params["emissions"]["rates"]))
print("True rates: {}".format(true_rates))
```

+++ {"id": "2E6o_kGKJ81Z"}

We see that the rates are the same for some states, which means those states are being treated as identical, and are therefore redundant. This is evidence of EM getting stuck in a local optimum.

+++ {"id": "fM_JX-feJ_pG"}

## Plotting the posterior over states

```{code-cell} ipython3
:id: d4qxA1cwgS1G

hmm_posterior = hmm.smoother(params, emissions)
posterior_probs = hmm_posterior.smoothed_probs
rates = params["emissions"]["rates"]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 687
id: oZ7C937t-Xh3
outputId: 9c4f663f-81f2-4b53-e8a6-0df0a5c89d3d
---
def plot_state_posterior(ax, state_posterior_probs, title):
    ln1 = ax.plot(state_posterior_probs, c="tab:blue", lw=3, label="p(state | counts)")
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("posterior probability")
    ax2 = ax.twinx()
    ln2 = ax2.plot(emissions, c="black", alpha=0.3, label="observed counts")
    ax2.set_title(title)
    ax2.set_xlabel("time")
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color="white")
    ax2.grid(False)


fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(2, 2, 1), posterior_probs[:, 0], title="state 0 (rate {:.2f})".format(rates[0,0]))
plot_state_posterior(fig.add_subplot(2, 2, 2), posterior_probs[:, 1], title="state 1 (rate {:.2f})".format(rates[1,0]))
plot_state_posterior(fig.add_subplot(2, 2, 3), posterior_probs[:, 2], title="state 2 (rate {:.2f})".format(rates[2,0]))
plot_state_posterior(fig.add_subplot(2, 2, 4), posterior_probs[:, 3], title="state 3 (rate {:.2f})".format(rates[3,0]))
plt.tight_layout()
```

```{code-cell} ipython3
:id: GSFx4aCb0lZU

# max marginals
most_probable_states = hmm.most_likely_states(params, emissions)
most_probable_rates = rates[most_probable_states]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 295
id: 84hUzqYn2Nky
outputId: 92f0298d-0d47-4fa6-b568-94c8dda286d6
---
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(most_probable_rates, c="tab:green", lw=3, label="inferred rate")
ax.plot(emissions, c="black", alpha=0.3, label="observed counts")
ax.set_ylabel("latent rate")
ax.set_xlabel("time")
ax.set_title("Inferred latent rate over time")
ax.legend(loc=4);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 295
id: g8kKmL5f2vAf
outputId: e02b7345-4856-4986-bf68-2645c259d06a
---
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
color_list = ["tab:red", "tab:green", "tab:blue", "k"]
colors = [color_list[z] for z in most_probable_states]
for i in range(len(colors)):
    ax.plot(i, most_probable_rates[i], "-o", c=colors[i], lw=3, alpha=0.75)
ax.plot(emissions, c="black", alpha=0.3, label="observed counts")
ax.set_ylabel("latent rate")
ax.set_xlabel("time")
ax.set_title("Inferred latent rate over time");
```

+++ {"id": "_RnpDkTKK4el"}

## Model with unknown $K$

+++ {"id": "pd-iztMNLBIb"}

In general we don't know the true number of states. One way to select the 'best' model is to compute the one with the maximum marginal likelihood. Rather than summing over both discrete latent states and integrating over the unknown parameters $\lambda$, we just maximize over the parameters (empirical Bayes approximation).

$$p(x_{1:T}|K) \approx \max_\lambda \int p(x_{1:T}, z_{1:T} | \lambda, K) dz$$
We can do this by fitting a bank of separate HMMs in parallel, one for each value of $K$. We need to make them all the same size so we can batch them efficiently. To do this, we pad the transition matrices (and other paraemeter vectors) so they all have the same shape, and then use masking.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 628zP3kx2hg6
outputId: 5b74ed10-bc40-4db0-8706-c17b04f16d70
---
max_num_states = 6
states = jnp.arange(1, max_num_states + 1)

# For each candidate model, build the initial state prior and transition matrix.
batch_initial_state_probs, batch_transition_probs = vmap(build_latent_state, in_axes=(0, None, None))(
    states, max_num_states, daily_change_prob
)

print("Shape of initial_state_probs: {}".format(batch_initial_state_probs.shape))
print("Shape of transition probs: {}".format(batch_transition_probs.shape))
print("Example initial state probs for num_states==3:\n{}".format(batch_initial_state_probs[2, :]))
print("Example transition_probs for num_states==3:\n{}".format(batch_transition_probs[2, :, :]))
```

+++ {"id": "kElabi3wjiRf"}

## Model fitting with gradient descent

```{code-cell} ipython3
:id: HqwITDc4emSB

hmm = NonconjugatePoissonHMM(max_num_states, emission_dim=1, 
                             emission_prior_loc=3.0,
                             emission_prior_scale=1.0)

def _fit(initial_probabilities, transition_matrix):
    # Set and freeze the initial distribution and transition matrix
    k1, k2 = jr.split(jr.PRNGKey(0), 2)
    params, param_props = hmm.random_initialization(k1)
    params["initial"]["probs"] = initial_probabilities
    params["transitions"]["transition_matrix"] = transition_matrix
    # params["emissions"]["rates"] = tfd.LogNormal(jnp.log(emissions.mean()), 0.25).sample(
    #     seed=k2, sample_shape=(max_num_states, 1))
    params["emissions"]["rates"] = jnp.array([3.0, 20.0, 40.0, 50.0, 30.0, 10.0]).reshape(max_num_states, 1)
    param_props["initial"]["probs"].trainable = False
    param_props["transitions"]["transition_matrix"].trainable = False
    
    optimizer = optax.adam(1e-1)
    num_epochs = 1000
    return hmm.fit_sgd(params, param_props, emissions[None, ...], optimizer=optimizer, num_epochs=num_epochs)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3rOAqjPdjLWV
outputId: 54d42375-5775-49ad-8046-ac8d0f4f7455
---
params, losses =vmap(_fit)(
    batch_initial_state_probs, 
    batch_transition_probs
)
rates = params["emissions"]["rates"]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 265
id: IuqR_1P5kBBX
outputId: 2ffdcc7c-fc89-4e83-cdb3-5f924b6a8475
---
plt.plot(losses.T)
plt.ylabel("Negative log marginal likelihood")
```

+++ {"id": "7FTjbb4Qj1pQ"}

## Plot marginal likelihood of each model

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 295
id: fe7CQTO9OuA8
outputId: fb00b663-bc67-4390-b4cf-7aff8fc90513
---
plt.plot(-losses[:, -1])
plt.ylabel("marginal likelihood $\\tilde{p}(x)$")
plt.xlabel("number of latent states")
plt.title("Model selection on latent states");
```

+++ {"id": "7aJLgl4NkegW"}

## Plot posteriors

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Lljaow1JOw1H
outputId: b16e8b7f-eaf1-46e7-be05-3f369e3be228
---
for i, rate in enumerate(rates):
    print("rates for {}-state model: {}".format(i + 1, rate))
```

```{code-cell} ipython3
:id: y7OEgXe6P5ro

most_probable_states = vmap(partial(hmm.most_likely_states, emissions=emissions))(params)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 315
id: r8aaKsNiRAxs
outputId: 61019282-a738-4db0-e3e8-14788037bcfa
---
fig = plt.figure(figsize=(14, 12))
for i, learned_model_rates in enumerate(rates):
    ax = fig.add_subplot(4, 3, i + 1)
    ax.plot(learned_model_rates[most_probable_states[i]], c="green", lw=3, label="inferred rate")
    ax.plot(emissions, c="black", alpha=0.3, label="observed counts")
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("{}-state model".format(i + 1))
    ax.legend(loc=4)
plt.tight_layout()
```
