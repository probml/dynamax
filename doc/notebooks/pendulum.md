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

+++ {"colab_type": "text", "id": "view-in-github"}

<a href="https://colab.research.google.com/github/petergchang/dynamax/blob/main/dynamax/nlgssm/demos/pendulum.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "4j2SH7ks97hb"}

# Pendulum Tracking Example for NLGSSM

+++ {"id": "AYrePRtV4RIK"}

This notebook demonstrates a simple pendulum tracking example for EKF, UKF, SLF, and PF using the implementations in `dynamax/nlgssm`.

The example is taken from Simo Särkkä (2013), “Bayesian Filtering and Smoothing,” Cambridge University Press. Available: https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf

The JAX translation is by Peter G. Chang ([@petergchang](https://github.com/petergchang)) and taken from the [särkkä-jax](https://github.com/petergchang/sarkka-jax) repo.


+++ {"id": "vLt4p8Tx4hWN"}

## 0. Imports and Formatting

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: djYX5weABagr
outputId: 0185bf5c-f9da-4ebf-fd5c-dd3e5068291f
---
try:
    import dynamax
except ModuleNotFoundError:
    %pip install git+https://github.com/probml/dynamax.git
    import dynamax
dynamax.__file__
```

```{code-cell} ipython3
:id: U4xUpwZM3-z5

%matplotlib inline
import matplotlib.pyplot as plt

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax import lax
from jax import jacfwd
from typing import Callable

from dynamax.linear_gaussian_ssm.inference import lgssm_filter
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from dynamax.extended_kalman_filter.inference import extended_kalman_filter, extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.containers import NLGSSMParams
```

```{code-cell} ipython3
:id: Lh8pVrU6oNsW

# For pretty print of ndarrays
jnp.set_printoptions(formatter={"float_kind": "{:.2f}".format})
```

+++ {"id": "33ISOI7QtWlP"}

## 1. Simulation and Plotting

+++ {"id": "BLuO9E_etVp-"}

We simulate the pendulum data using the following transition model:
\begin{align*}
  \begin{pmatrix} x_{1,k} \\ x_{2,k} \end{pmatrix} &= 
  \begin{pmatrix} x_{1,k-1} + x_{2,k-1} \Delta t \\
    x_{2,k-1} - g \sin(x_{1,k-1}) \Delta t
  \end{pmatrix} + q_{k-1} \\
  y_{k} &= \sin(x_{1,k}) + r_k \\
  y_{x_k} &= \arcsin(y_k) = \arcsin(\sin(x_{1,k}) + r_k)
\end{align*}

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cZftka6dazaA
outputId: 795747e0-264b-4fb3-e1cb-fef7e85cc0c4
---
# Some parameters
dt = 0.0125
g = 9.8
q_c = 1
r = 0.3

# Lightweight container for pendulum parameters
@chex.dataclass
class PendulumParams:
    initial_state: chex.Array = jnp.array([jnp.pi / 2, 0])
    dynamics_function: Callable = lambda x: jnp.array([x[0] + x[1] * dt, x[1] - g * jnp.sin(x[0]) * dt])
    dynamics_covariance: chex.Array = jnp.array([[q_c * dt**3 / 3, q_c * dt**2 / 2], [q_c * dt**2 / 2, q_c * dt]])
    emission_function: Callable = lambda x: jnp.array([jnp.sin(x[0])])
    emission_covariance: chex.Array = jnp.eye(1) * (r**2)
```

```{code-cell} ipython3
:id: GTcJutobtduz

# Pendulum simulation (Särkkä Example 3.7)
def simulate_pendulum(params=PendulumParams(), key=0, num_steps=400):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    # Unpack parameters
    M, N = params.initial_state.shape[0], params.emission_covariance.shape[0]
    f, h = params.dynamics_function, params.emission_function
    Q, R = params.dynamics_covariance, params.emission_covariance

    def _step(carry, rng):
        state = carry
        rng1, rng2 = jr.split(rng, 2)

        next_state = f(state) + jr.multivariate_normal(rng1, jnp.zeros(M), Q)
        obs = h(next_state) + jr.multivariate_normal(rng2, jnp.zeros(N), R)
        return next_state, (next_state, obs)

    rngs = jr.split(key, num_steps)
    _, (states, observations) = lax.scan(_step, params.initial_state, rngs)
    return states, observations


states, obs = simulate_pendulum()
```

```{code-cell} ipython3
:id: 4-sZDJ0odC8d

# Helper function for plotting
def plot_pendulum(time_grid, x_tr, x_obs, x_est=None, est_type=""):
    plt.figure()
    plt.plot(time_grid, x_tr, color="darkgray", linewidth=4, label="True Angle")
    plt.plot(time_grid, x_obs, "ok", fillstyle="none", ms=1.5, label="Measurements")
    if x_est is not None:
        plt.plot(time_grid, x_est, color="k", linewidth=1.5, label=f"{est_type} Estimate")
    plt.xlabel("Time $t$")
    plt.ylabel("Pendulum angle $x_{1,k}$")
    plt.xlim(0, 5)
    plt.ylim(-3, 5)
    plt.xticks(jnp.arange(0.5, 4.6, 0.5))
    plt.yticks(jnp.arange(-3, 5.1, 1))
    plt.gca().set_aspect(0.5)
    plt.legend(loc=1, borderpad=0.5, handlelength=4, fancybox=False, edgecolor="k")
    plt.show()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 285
id: QTTt7NKfwPwR
outputId: 71d45ddc-6e06-4e46-ebd4-96ea05c4b0af
---
# Create time grid for plotting
time_grid = jnp.arange(0.0, 5.0, step=dt)

# Plot the generated data
plot_pendulum(time_grid, states[:, 0], obs)
```

```{code-cell} ipython3
:id: Fb3OXaaEiEoV

# Compute RMSE
def compute_rmse(y, y_est):
    return jnp.sqrt(jnp.sum((y - y_est) ** 2) / len(y))


# Compute RMSE of estimate and print comparison with
# standard deviation of measurement noise
def compute_and_print_rmse_comparison(y, y_est, R, est_type=""):
    rmse_est = compute_rmse(y, y_est)
    print(f'{f"The RMSE of the {est_type} estimate is":<40}: {rmse_est:.2f}')
    print(f'{"The std of measurement noise is":<40}: {jnp.sqrt(R):.2f}')
```

+++ {"id": "5Rx9XE5Af2sP"}

## 2. Extended Kalman Filter

```{code-cell} ipython3
:id: jWRhHTOCgIAj

pendulum_params = PendulumParams()

# Define parameters for EKF
ekf_params = NLGSSMParams(
    initial_mean=pendulum_params.initial_state,
    initial_covariance=jnp.eye(states.shape[-1]) * 0.1,
    dynamics_function=pendulum_params.dynamics_function,
    dynamics_covariance=pendulum_params.dynamics_covariance,
    emission_function=pendulum_params.emission_function,
    emission_covariance=pendulum_params.emission_covariance,
)

# Compute EKF estimate using SSM Library
ekf_posterior = extended_kalman_filter(ekf_params, obs)
m_ekf = ekf_posterior.filtered_means[:, 0]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 285
id: icfpJJs3ik2i
outputId: a068fcc5-f67e-494a-ad21-c868bcefd80b
---
# Plot EKF results
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_ekf, est_type="EKF")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FcUBT4DSUDgo
outputId: 4b4b23c8-4a81-4759-aee8-5776e32e6923
---
compute_and_print_rmse_comparison(states[:, 0], m_ekf, r, "EKF")
```

+++ {"id": "8uUJWaZ0zC8g"}

#3. Extended Kalman Smoother

```{code-cell} ipython3
:id: o8yuQrnAzJKu

# Compute EK smoothed estimate using SSM Library
ekf_smoothed_posterior = extended_kalman_smoother(ekf_params, obs)
m_ekf_smoothed = ekf_smoothed_posterior.smoothed_means[:, 0]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 285
id: pX1pBHSJzQkn
outputId: 5422377e-64c8-4bb9-90b8-858d87fca64a
---
# Plot EKF results
plot_pendulum(time_grid, states[:, 0], obs, x_est=m_ekf_smoothed, est_type="EKF")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: rHssvE-czT1X
outputId: 37f60ad1-d589-457f-a561-4df6695bd0e1
---
compute_and_print_rmse_comparison(states[:, 0], m_ekf_smoothed, r, "EKF")
```
