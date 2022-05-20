import numpy as np
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from ssm_jax.demos.plot_utils import plot_ellipse
from ssm_jax.lgssm.inference import LGSSMParams, lgssm_sample, lgssm_filter, lgssm_smoother

def plot_tracking_values(observed, filtered, cov_hist, signal_label, ax):
    """
    observed: array(nsteps, 2)
        Array of observed values
    filtered: array(nsteps, state_size)
        Array of latent (hidden) values. We consider only the first
        two dimensions of the latent values
    cov_hist: array(nsteps, state_size, state_size)
        History of the retrieved (filtered) covariance matrices
    ax: matplotlib AxesSubplot
    """
    timesteps, _ = observed.shape
    ax.plot(observed[:, 0], observed[:, 1], marker="o", linewidth=0,
            markerfacecolor="none", markeredgewidth=2, markersize=8, label="observed", c="tab:green")
    ax.plot(*filtered[:, :2].T, label=signal_label, c="tab:red", marker="x", linewidth=2)
    for t in range(0, timesteps, 1):
        covn = cov_hist[t][:2, :2]
        plot_ellipse(covn, filtered[t, :2], ax, n_std=2.0, plot_center=False)
    ax.axis("equal")
    ax.legend()

def plot_state_space(z, ax=None, **plt_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    x,y = z[:,:2].T
    if 'marker' not in plt_kwargs:
        plt_kwargs['marker'] = "o"
    ax.plot(x,y,**plt_kwargs)
    ax.axis('equal');


delta = 1.0
F = jnp.array([
    [1, 0, delta, 0],
    [0, 1, 0, delta],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


H = jnp.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

state_size, _ = F.shape
observation_size, _ = H.shape

G = jnp.zeros((state_size,1))
J = jnp.zeros((observation_size,1))


Q = jnp.eye(state_size) * 0.001
R = jnp.eye(observation_size) * 1.0

# Prior parameter distribution
mu0 = jnp.array([8, 10, 1, 0]).astype(float)
Sigma0 = jnp.eye(state_size) * 0.1

lgssm = LGSSMParams(initial_mean = mu0,
                    initial_covariance = Sigma0,
                    dynamics_matrix = F,
                    dynamics_input_weights = G,
                    dynamics_covariance = Q,
                    emission_matrix = H,
                    emission_input_weights = J,
                    emission_covariance = R)

key = jr.PRNGKey(111)
num_timesteps = 15
inputs = jnp.zeros((num_timesteps,1))

x, y = lgssm_sample(key,lgssm,num_timesteps,inputs)

ll_filt, filtered_means, filtered_covs = lgssm_filter(lgssm, inputs, y)
ll_smooth, smoothed_means, smoothed_covs, _ = lgssm_smoother(lgssm, inputs, y)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
plot_tracking_values(y,filtered_means,filtered_covs, "filtered", ax1)
plot_tracking_values(y,smoothed_means,smoothed_covs, "smoothed", ax2)
plt.show()
