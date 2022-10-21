---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.8.14 64-bit
  language: python
  name: python3
---

# Causal Impact Jax

+++

The causal impact R package: <a href="https://google.github.io/CausalImpact/CausalImpact.html"> CausalImpact</a> is built upon the R package <a href="https://cran.r-project.org/web/packages/bsts/bsts.pdf"> bsts </a> for Bayesian structural time series models.

There are some python packages that implement the causal impact algorithm with python:
* The package <a href="https://github.com/WillianFuks/tfcausalimpact"> tfcausalimpact </a> is built upon the 
<a href="https://www.tensorflow.org/probability/api_docs/python/tfp/sts"> tfp.sts module </a>.
* The package <a href="https://pypi.org/project/pycausalimpact/"> pycausalimpact </a> is built upon the package
 <a href="https://github.com/statsmodels/statsmodels"> statsmodels </a>.
* The package <a href="https://github.com/jamalsenouci/causalimpact"> causalimpact </a> is also built upon the package ‘statsmodel’.

The R package ‘bsts’, python packages ‘tfp.sts’, and ‘statsmodels’ all contain functions that build the structural time series model and perform posterior inference and forecast predictions.

+++

## Causal impact

+++

## Example 

Use the same example as in the R package <a href="http://google.github.io/CausalImpact/CausalImpact.html"> CausalImpact</a>.

```{code-cell} ipython3
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import matplotlib.pyplot as plt

from causal_impact import causal_impact
```

```{code-cell} ipython3
num_timesteps = 100

def auto_regress(x, key):
    x_new = 0.99 * x + jr.normal(key)
    return x_new, x

key = jr.PRNGKey(1)
x0 = jr.normal(key)
keys = jr.split(key, num_timesteps)
_, x =  lax.scan(auto_regress, x0, keys)
x = x + 100

y = 1.2*x + jr.normal(key, shape=(num_timesteps,))
y = y.at[70:].set(y[70:]+10)

plt.plot(y, label='Y', color='black', lw=2)
plt.plot(x, linestyle='dashed', color='red', lw=2, label='X')
plt.legend()
```

```{code-cell} ipython3
# Run an anlysis
observed_timeseries = jnp.expand_dims(y, 1)
inputs = jnp.concatenate((jnp.ones((num_timesteps, 1)), jnp.expand_dims(x, 1)), axis=1)
intervention_time = 70

impact = causal_impact(observed_timeseries, intervention_time, 'Gaussian', inputs,
                       sts_model=None, confidence_level=0.95, key=jr.PRNGKey(0), sample_size=200)
```

```{code-cell} ipython3
impact.plot()
```
