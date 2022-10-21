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

<a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book2/28/kf_spiral.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "f9fe3938"}

# Tracking a 2d point spiraling in the plane using the Kalman filter

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: CZGgVBb0iMIa
outputId: 9e8742b5-7b8a-46aa-d521-d8f4a59a52e2
---
%pip install -qq git+https://github.com/probml/dynamax.git
```

```{code-cell}
:id: zOGCXPVjiPBj

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
  height: 830
id: dYzP_ksJiTrZ
outputId: bbcb51f3-00c1-4924-eee8-d52db7d4678e
---
try:
    from dynamax.linear_gaussian_ssm.demos.kf_spiral import kf_spiral, plot_kf_spiral
except ModuleNotFoundError:
    %pip install -qq dynamax
    from dynamax.linear_gaussian_ssm.demos.kf_spiral import kf_spiral, plot_kf_spiral

x, y, lgssm_posterior = kf_spiral()
dict_figures = plot_kf_spiral(x, y, lgssm_posterior)

for k, v in dict_figures.items():
    fname = k + ".pdf"
    print(fname)
    fig = v
    fig.savefig(fname)
```
