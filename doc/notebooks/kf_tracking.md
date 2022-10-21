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

+++ {"id": "92e34046"}

# Tracking a 2d point moving in the plane using the Kalman filter

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: iCSsZdpXLme0
outputId: bf28b24b-0476-4b9d-95e6-99b3a40849a2
---


try:
    import dynamax
except ModuleNotFoundError:
    print('installing dynamax')
    %pip install -qq git+https://github.com/probml/dynamax.git
    import dynamax
```

```{code-cell} ipython3
:id: KFpcV2mCL1DN

# Silence WARNING:root:The use of `check_types` is deprecated and does not have any effect.
# https://github.com/tensorflow/probability/issues/1523
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 813
id: nR-8u6MlhTWQ
outputId: 0263be0e-cf8c-44a2-da65-0f65c037b27d
---
from dynamax.linear_gaussian_ssm.demos.kf_tracking import kf_tracking, plot_kf_tracking

x, y, lgssm_posterior = kf_tracking()
dict_figures = plot_kf_tracking(x, y, lgssm_posterior)

for k, v in dict_figures.items():
    fname = k + ".pdf"
    print('saving ', fname)
    fig = v
    fig.savefig(fname)
```
