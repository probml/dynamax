#!/usr/bin/env python
# coding: utf-8

# # Tracking a 2d point moving in the plane using the Kalman filter
# 
# We use the [dynamax](https://github.com/probml/dynamax/blob/main/dynamax/) library.
# 

# In[1]:




try:
    import dynamax
except ModuleNotFoundError:
    print('installing dynamax')
    get_ipython().run_line_magic('pip', 'install -qq git+https://github.com/probml/dynamax.git')
    import dynamax


# In[2]:


# Silence WARNING:root:The use of `check_types` is deprecated and does not have any effect.
# https://github.com/tensorflow/probability/issues/1523
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


# In[3]:


from dynamax.linear_gaussian_ssm.demos.kf_tracking import kf_tracking, plot_kf_tracking

x, y, lgssm_posterior = kf_tracking()
dict_figures = plot_kf_tracking(x, y, lgssm_posterior)

for k, v in dict_figures.items():
    fname = k + ".pdf"
    print('saving ', fname)
    fig = v
    fig.savefig(fname)

