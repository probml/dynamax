---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: 'Python 3.10.6 (''.venv'': venv)'
  language: python
  name: python3
---

+++ {"id": "AvMD00M4GzLg"}

# 0. Imports

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: q702_aHMbeSL
outputId: 41ebf366-c64d-4ed4-c51c-c6e5545dedbf
---
try:
    from dynamax.cond_moments_gaussian_filter.optimizer import CMGFOptimizerParams, CMGFOptimizer
    from dynamax.cond_moments_gaussian_filter.demos.cmgf_mlp_classifier import *
except ModuleNotFoundError:
    print('installing dynamax')
    %pip install -qq git+https://github.com/probml/dynamax.git
    from dynamax.cond_moments_gaussian_filter.optimizer import CMGFOptimizerParams, CMGFOptimizer
    from dynamax.cond_moments_gaussian_filter.demos.cmgf_mlp_classifier import *
```

```{code-cell} ipython3
:id: Mu2BQ1ElG773

from functools import partial

import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
import optax
```

+++ {"id": "nCM-lh50-Aw4"}

# 1. MLP Classifier Example

+++ {"id": "cAhWgsGlc_II"}

We set up some groundwork.

```{code-cell} ipython3
:id: eLe13WgRdEtV

def evaluate_and_plot(ax, input_grid, params, predict_fn, input, output, title="CMGF-EKF One-Pass Trained MLP Classifier"):
    # Evaluate the trained MLP on grid and plot
    Z_cmgf = posterior_predictive_grid(input_grid, params, predict_fn, binary=False)
    return plot_posterior_predictive(ax, input, output, title, input_grid, Z_cmgf)
```

```{code-cell} ipython3
:id: TbJ0yMy5BD_m

# Generate spiral dataset
input, output = generate_spiral_dataset()

# Define grid on input space
input_grid = generate_input_grid(input)
```

```{code-cell} ipython3
:id: 5jQEVTiL1XBA

# Define MLP architecture
input_dim, hidden_dims, output_dim = 2, [15, 15], 1
model_dims = [input_dim, *hidden_dims, output_dim]
_, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

# Define model and initial opt_state
state_dim, emission_dim = flat_params.size, output_dim
eps = 1e-4
sigmoid_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(apply_fn(w, x)), eps, 1-eps) # Clip to prevent divergence
pred_mean_fn = lambda w, x: sigmoid_fn(w, x)
pred_cov_fn = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))
initial_mean=flat_params
initial_covariance=jnp.eye(state_dim)
```

```{code-cell} ipython3
:id: ICZ6SeVC_0YI

def fit(opt_state, optimizer, input, output, pred_mean_fn, pred_cov_fn):
    params = opt_state.mean

    @jit
    def step(params, opt_state, x, y):
        updates, opt_state = optimizer.update(x, pred_mean_fn, pred_cov_fn, y, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    for i, (x, y) in enumerate(zip(input, output)):
        params, opt_state = step(params, opt_state, x, y)
    
    return params
```

+++ {"id": "zCo95mlZdeft"}

Next, we try some experiments.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: L1wnw_ACd9nq
outputId: dc93cee3-5a66-4c02-9442-9e4b66adbeeb
---
# Initialize cmgf-ekf optimizer
ekf_optimizer = CMGFOptimizer('ekf')
opt_state = CMGFOptimizerParams(mean=initial_mean, cov=initial_covariance)
params = fit(opt_state, ekf_optimizer, input, output, pred_mean_fn, pred_cov_fn)

fig, ax = plt.subplots(figsize=(6, 5))
evaluate_and_plot(ax, input_grid, params, sigmoid_fn, input, output);
```

+++ {"id": "_kNygINkdzHZ"}

Next we simplify the model to decrease the predictive power.

```{code-cell} ipython3
:id: yWKlS5huevn-

# Define MLP architecture
input_dim, hidden_dims, output_dim = 2, [7, 7], 1
model_dims = [input_dim, *hidden_dims, output_dim]
_, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

# Define model and initial opt_state
state_dim, emission_dim = flat_params.size, output_dim
eps = 1e-4
sigmoid_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(apply_fn(w, x)), eps, 1-eps) # Clip to prevent divergence
pred_mean_fn = lambda w, x: sigmoid_fn(w, x)
pred_cov_fn = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))
initial_mean=flat_params
initial_covariance=jnp.eye(state_dim)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: TvVxYHUUeyE5
outputId: 0278bea8-83c1-4906-9301-2532d26f0224
---
# Initialize cmgf-ekf optimizer
ekf_optimizer = CMGFOptimizer('ekf')
opt_state = CMGFOptimizerParams(mean=initial_mean, cov=initial_covariance)
params = fit(opt_state, ekf_optimizer, input, output, pred_mean_fn, pred_cov_fn)

fig, ax = plt.subplots(figsize=(6, 5))
evaluate_and_plot(ax, input_grid, params, sigmoid_fn, input, output);
```

+++ {"id": "rGdtrNyBfAuX"}

Note below that iterated posterior linearization is able to make better predictions despite the limiting simplicity of the model. However, training takes longer.

```{code-cell} ipython3
:id: ZE5l5ZSFDKvb

# Initialize cmgf optimizer
ekf_optimizer = CMGFOptimizer('ekf', 40)
opt_state = CMGFOptimizerParams(mean=initial_mean, cov=initial_covariance)
params = fit(opt_state, ekf_optimizer, input, output, pred_mean_fn, pred_cov_fn)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: dVO3dvywaN1b
outputId: 7a7a04fd-f975-4530-efd7-a8fd4997a40e
---
fig, ax = plt.subplots(figsize=(6, 5))
evaluate_and_plot(ax, input_grid, params, sigmoid_fn, input, output);
```

+++ {"id": "Fz3ob9jFKC7P"}

# 3. Comparison with SGD

+++ {"id": "6jKEycjKRjGZ"}

For comparison, we move back to the more complex MLP architecture.

```{code-cell} ipython3
:id: 8Jzw7SsJRio-

# Define MLP architecture
input_dim, hidden_dims, output_dim = 2, [15, 15], 1
model_dims = [input_dim, *hidden_dims, output_dim]
_, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

# Define model and initial opt_state
state_dim, emission_dim = flat_params.size, output_dim
eps = 1e-4
sigmoid_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(apply_fn(w, x)), eps, 1-eps) # Clip to prevent divergence
pred_mean_fn = lambda w, x: sigmoid_fn(w, x)
pred_cov_fn = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))
initial_mean=flat_params
initial_covariance=jnp.eye(state_dim)
```

```{code-cell} ipython3
:id: myrBnBdWKpo8

def loss_optax(params, x, y, apply_fn):
    y_hat = apply_fn(params, x)
    loss_value = -(y * jnp.log(y_hat) + (1-y) * jnp.log(1 - y_hat))
    return loss_value.mean()
```

```{code-cell} ipython3
:id: baRer9WNLtiH

loss_fn = partial(loss_optax, apply_fn = sigmoid_fn)
```

```{code-cell} ipython3
:id: Tm65xOD6KGsw

def fit_optax(params, optimizer, input, output, loss_fn, num_epochs):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(input, output)):
            params, opt_state, loss_value = step(params, opt_state, x, y)
    
    return params
```

```{code-cell} ipython3
:id: dN2CvZ-LMQZa

sgd_optimizer = optax.sgd(learning_rate=1e-2)
```

```{code-cell} ipython3
:id: unn8ZAXoMYrx

params = fit_optax(flat_params, sgd_optimizer, input, output, loss_fn, num_epochs=1)
```

+++ {"id": "5gv03kveRqg-"}

Single-pass SGD is no good.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: 9vkBccIeMk-s
outputId: 3488986a-6152-4177-a0ef-f3dde95dfc4a
---
fig, ax = plt.subplots(figsize=(6, 5))
evaluate_and_plot(ax, input_grid, params, sigmoid_fn, input, output);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: nJ7MIx19SDS-
outputId: 088b8e63-28e4-4525-b7c1-fc892860ab0f
---
epoch_range = jnp.arange(1, 30, 3)
n = len(epoch_range)
fig, ax = plt.subplots(n // 2, 2, figsize=(8, 16))
for step, axi in zip(epoch_range, ax.flatten()):
    params = fit_optax(flat_params, sgd_optimizer, input, output, loss_fn, num_epochs=step)
    title = f'SGD training with {step} epochs'
    evaluate_and_plot(axi, input_grid, params, sigmoid_fn, input, output, title=title)
```
