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

+++ {"id": "8a63fcca"}



# The "occasionally dishonest casino" HMM

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: zRceKxSmjjaX
outputId: 25dd3aa0-a2d0-4706-d37b-e6caa9197c88
---
try:
    import probml_utils as pml
    from probml_utils import savefig, latexify
except ModuleNotFoundError:
    %pip install -qq git+https://github.com/probml/probml-utils.git
    import probml_utils as pml
    from probml_utils import savefig, latexify
```

```{code-cell} ipython3
try:
    from dynamax.hmm.demos.casino_hmm import make_model_and_data, plot_results, plot_inference
except ModuleNotFoundError:
    %pip install -qq dynamax
    from dynamax.hmm.demos.casino_hmm import make_model_and_data, plot_results, plot_inference
```

```{code-cell} ipython3

from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import optax

from graphviz import Digraph
import matplotlib.pyplot as plt

from dynamax.hmm.models import CategoricalHMM
```

```{code-cell} ipython3
latexify(width_scale_factor=3.2, fig_height=1.5)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 990
id: t9ZrTumQjsLN
outputId: 26626071-8890-4ce4-dbc2-5378ddac4e67
---
num_states = 2
num_emissions = 1
num_classes = 6
num_batches = 1
num_timesteps = 300
hmm = CategoricalHMM(num_states, num_emissions, num_classes)

params = dict(
    initial=dict(probs=jnp.array([1, 1]) / 2),
    transitions=dict(transition_matrix=jnp.array([[0.95, 0.05], [0.10, 0.90]])),
    emissions=dict(probs=jnp.array(
        [
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],  # fair die
            [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 5 / 10],  # loaded die
        ]
    ).reshape(num_states, num_emissions, num_classes))
)

batch_states, batch_emissions = \
    vmap(partial(hmm.sample, params, num_timesteps=num_timesteps))(
        jr.split(jr.PRNGKey(0), num_batches))
```

```{code-cell} ipython3
posterior = hmm.smoother(params, batch_emissions[0])
most_likely_states = hmm.most_likely_states(params, batch_emissions[0])
```

```{code-cell} ipython3
def print_seq(true_states, emissions):
    T = 70
    to_string = lambda x: "".join(str(x + 1).split())
    print("hid: ", to_string(true_states[:T]))
    print("obs: ", to_string(emissions[:T, 0]))
    
print_seq(batch_states[0], batch_emissions[0])
```

```{code-cell} ipython3
# count fraction of times we see 6 in each state
obs = batch_emissions[0, :, 0] + 1
hid = batch_states[0] + 1 
p0 = jnp.mean(obs[hid==1] == 6) # fair
p1 = jnp.mean(obs[hid==2] == 6) # loaded
print(jnp.array([p0, p1]))
print([1.0/6, 5.0/10])
```

```{code-cell} ipython3
:id: e223JBC5DhPP

def plot_posteriors(true_states, emissions, posterior, most_likely_states):

    fig, ax = plt.subplots()
    plot_inference(posterior.filtered_probs, true_states, ax)
    ax.set_ylabel("p(loaded)")
    ax.set_title("Filtered")
    fig.show()
    # pml.savefig("hmm_casino_filter")

    fig, ax = plt.subplots()
    plot_inference(posterior.smoothed_probs, true_states, ax)
    ax.set_ylabel("p(loaded)")
    ax.set_title("Smoothed")
    fig.show()
    # pml.savefig("hmm_casino_smooth")

    fig, ax = plt.subplots()
    plot_inference(most_likely_states, true_states, ax, map_estimate=True)
    ax.set_ylabel("MAP state")
    ax.set_title("Viterbi")
    fig.show()
    # pml.savefig("hmm_casino_map")

plot_posteriors(batch_states[0], batch_emissions[0], posterior, most_likely_states)
```
