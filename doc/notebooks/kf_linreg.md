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

<a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book2/28/kf_linreg.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "FwspQ4wDgnp3"}

# Recursive least squares using Kalman filtering

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.

```{code-cell}
:id: sO8k0Mi4gv9U

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
  height: 314
id: 9nkWIAnGgwJi
outputId: 69a2ba91-9084-4c5f-e318-98def553c8ad
---
try:
    from dynamax.linear_gaussian_ssm.demos.kf_linreg import *
except ModuleNotFoundError:
    %pip install -qq git+https://github.com/probml/dynamax.git
    from dynamax.linear_gaussian_ssm.demos.kf_linreg import *


kf_results, batch_results = online_kf_vs_batch_linreg()
dict_figures = plot_online_kf_vs_batch_linreg(kf_results, batch_results)


for k, v in dict_figures.items():
    fname = k + ".pdf"
    print(fname)
    fig = v
    fig.savefig(fname)

plt.show()
```
