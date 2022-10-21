---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.9.6
  language: python
  name: python3
---

```{code-cell} ipython3
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

from ssm_jax.hmm.models import MultivariateNormalDiagPlusLowRankHMM
```

```{code-cell} ipython3
hmm = MultivariateNormalDiagPlusLowRankHMM.random_initialization(jr.PRNGKey(0), 3, 2, 1)
hmm.emission_cov_diag_factors.value = 0.1 * jnp.ones((3, 2))
```

```{code-cell} ipython3
hmm.emission_cov_low_rank_factors.value.shape
```

```{code-cell} ipython3
states, emissions = hmm.sample(jr.PRNGKey(0), 1000)
```

```{code-cell} ipython3
for k in range(hmm.num_states):
    plt.plot(emissions[states==k, 0], emissions[states==k, 1], '.')
```

```{code-cell} ipython3
test_hmm = MultivariateNormalDiagPlusLowRankHMM.random_initialization(jr.PRNGKey(1), 3, 2, 1)
```

```{code-cell} ipython3
lps = test_hmm.fit_em(jnp.expand_dims(emissions, 0))
```

```{code-cell} ipython3
plt.plot(lps)
```

```{code-cell} ipython3
new_states, new_emissions = test_hmm.sample(jr.PRNGKey(2), 1000)

```

```{code-cell} ipython3
for k in range(test_hmm.num_states):
    plt.plot(new_emissions[new_states==k, 0], new_emissions[new_states==k, 1], '.')
```
