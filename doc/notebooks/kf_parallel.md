---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: 'Python 3.9.12 (''venv'': venv)'
  language: python
  name: python3
---

+++ {"colab_type": "text", "id": "view-in-github"}

<a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book2/28/kf_parallel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "b9420eab"}

# Tracking multiple 2d points moving in the plane using the Kalman filter

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: eaku_0reis3c
outputId: 354abd6f-863e-40f3-cd2c-56413348177f
---
%pip install -qq git+https://github.com/probml/dynamax.git
```

```{code-cell}
:id: 1EHgWmPtiy_k

# Silence WARNING:root:The use of `check_types` is deprecated and does not have any effect.
# https://github.com/tensorflow/probability/issues/1523
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 878
id: iVNKXcLLi2vI
outputId: f9014c25-ff65-413d-8b40-722930ee9348
---
try:
    from dynamax.linear_gaussian_ssm.demos.kf_parallel import kf_parallel, plot_kf_parallel
except ModuleNotFoundError:
    %pip install -qq dynamax
    from dynamax.linear_gaussian_ssm.demos.kf_parallel import kf_parallel, plot_kf_parallel

x, y, lgssm_posterior = kf_parallel()
dict_figures = plot_kf_parallel(x, y, lgssm_posterior)

for k, v in dict_figures.items():
    fname = k + ".pdf"
    print(fname)
    fig = v
    fig.savefig(fname)
```
