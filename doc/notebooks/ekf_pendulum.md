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

+++ {"id": "6qUUy7kVjUs4"}

# Pendulum tracking using extended Kalman filter

We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.

```{code-cell} ipython3
:id: luMgNLVOjQMg

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
:id: 6wjltvn7jd-I

%reload_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lKAlX02VjfK0
outputId: 3f52934a-634d-49ee-a71f-4fcc8010cc08
---
  # https://github.com/probml/probml-utils/blob/main/probml_utils/plotting.py\n",
try:
    import probml_utils as pml
except ModuleNotFoundError:
    print('installing probml_utils')
    %pip install -qq git+https://github.com/probml/probml-utils.git
    import probml_utils as pml
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ofnbsYpGjmyy
outputId: fe6d6860-3977-4e61-ab7a-fe828c7b353a
---
try:
    from dynamax.extended_kalman_filter.demos.ekf_pendulum import *
except ModuleNotFoundError:
    print('installing dynamax')
    %pip install -qq git+https://github.com/probml/dynamax.git
    from dynamax.extended_kalman_filter.demos.ekf_pendulum import *
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 761
id: PwdlwUZHjpJj
outputId: d6d92fe7-8f8e-43e7-8428-d1d26d00608b
---
pml.latexify(fig_width=2, fig_height=1.5) # figure size in inches for book
import matplotlib.pyplot as plt

dict_figures = main()
for fname, fig in dict_figures.items():
    #fig.savefig(fname)
    plt.figure(num=fig.number)
    pml.savefig(fname)
plt.show()
```
