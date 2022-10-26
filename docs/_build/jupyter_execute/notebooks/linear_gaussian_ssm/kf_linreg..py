#!/usr/bin/env python
# coding: utf-8

# # Online linear regression using Kalman filtering
# 
# We perform sequential (recursive) Bayesian inference for the parameters of a linear regression model
# using the Kalman filter. (This algorithm is also known as recursive least squares.)
# To do this, we treat the parameers of the model as the unknown hidden states.
# We assume that these are constant over time.
# The graphical model is shown below.
# 
# ![RLS](https://github.com/probml/dynamax/blob/main/docs/figures/rlsDgm.png?raw=true)
# 
# The model has the following form
# 
# \begin{align}
# \theta_t &=  \theta_{t-1}  \\
# y_t &= x_t^T \theta_t + r_t, \; r_t \sim N(0, \sigma^2)
# \end{align}
# 
# This is a LG-SSM, where $F=I$, $Q=0$, $H_t = x_t^T$ and $R = \sigma^2$.
# 

# # Setup

# In[1]:


from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


try:
    import dynamax
except ModuleNotFoundError:
    print('installing dynamax')
    get_ipython().run_line_magic('pip', 'install -qq git+https://github.com/probml/dynamax.git')
    import dynamax


from dynamax.linear_gaussian_ssm.inference import lgssm_filter, LGSSMParams


# In[3]:


# Silence WARNING:root:The use of `check_types` is deprecated and does not have any effect.
# https://github.com/tensorflow/probability/issues/1523
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


# # Data
# 
# Data is from https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m
# 

# In[4]:


n_obs = 21
x = jnp.linspace(0, 20, n_obs)
X = jnp.column_stack((jnp.ones_like(x), x))  # Design matrix.
y = jnp.array(
    [2.486, -0.303, -4.053, -4.336, -6.174, -5.604, -3.507, -2.326, -4.638, -0.233, -1.986, 1.028, -2.264,
     -0.451, 1.167, 6.652, 4.145, 5.268, 6.34, 9.626, 14.784])

#print([np.round(v, decimals=3) for v in np.array(y)])


# # Model

# In[5]:


F = jnp.eye(2)
Q = jnp.zeros((2, 2))  # No parameter drift.
obs_var = 1.0
R = jnp.ones((1, 1)) * obs_var
mu0 = jnp.zeros(2)
Sigma0 = jnp.eye(2) * 10.0

lgssm = LGSSMParams(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=X[:, None, :], # (t, 1, D) where D = num input features
        emission_covariance=R,
    )


# # Online inference

# In[6]:


lgssm_posterior = lgssm_filter(lgssm, y[:, None]) # reshape y to be (T,1)


kf_results = (lgssm_posterior.filtered_means, lgssm_posterior.filtered_covariances)


# # Offline inferenece
# 
# We compute the offline posterior given all the data using Bayes rule for linear regression.
# This should give the same results as the final step of online inference.

# In[7]:


posterior_prec = jnp.linalg.inv(Sigma0) + X.T @ X / obs_var
b = jnp.linalg.inv(Sigma0) @ mu0 + X.T @ y / obs_var
posterior_mean = jnp.linalg.solve(posterior_prec, b)
batch_results = (posterior_mean, posterior_prec)


# # Plot results

# In[8]:


# Unpack kalman filter results
post_weights_kf, post_sigma_kf = kf_results
w0_kf_hist, w1_kf_hist = post_weights_kf.T
w0_kf_err, w1_kf_err = jnp.sqrt(post_sigma_kf[:, [0, 1], [0, 1]].T)

# Unpack batch results
post_weights_batch, post_prec_batch = batch_results
w0_post_batch, w1_post_batch = post_weights_batch
Sigma_post_batch = jnp.linalg.inv(post_prec_batch)
w0_std_batch, w1_std_batch = jnp.sqrt(Sigma_post_batch[[0, 1], [0, 1]])

fig, ax = plt.subplots()
timesteps = jnp.arange(len(w0_kf_hist))

# Plot online kalman filter posterior.
ax.errorbar(timesteps, w0_kf_hist, w0_kf_err, fmt="-o", label="$w_0$", color="black", fillstyle="none")
ax.errorbar(timesteps, w1_kf_hist, w1_kf_err, fmt="-o", label="$w_1$", color="tab:red")

# Plot batch posterior.
ax.hlines(y=w0_post_batch, xmin=timesteps[0], xmax=timesteps[-1], color="black", label="$w_0$ batch")
ax.hlines(
    y=w1_post_batch, xmin=timesteps[0], xmax=timesteps[-1], color="tab:red", linestyle="--", label="$w_1$ batch"
)
ax.fill_between(timesteps, w0_post_batch - w0_std_batch, w0_post_batch + w0_std_batch, color="black", alpha=0.4)
ax.fill_between(timesteps, w1_post_batch - w1_std_batch, w1_post_batch + w1_std_batch, color="tab:red", alpha=0.4)

ax.set_xlabel("time")
ax.set_ylabel("weights")
ax.legend()


# # Run the script

# In[9]:



from dynamax.linear_gaussian_ssm.demos.kf_linreg import *


kf_results, batch_results = online_kf_vs_batch_linreg()
dict_figures = plot_online_kf_vs_batch_linreg(kf_results, batch_results)

plt.show()

