from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

from ssm_jax.linear_gaussian_ssm.inference_test import lgssm_ssm_jax_to_tfp
from ssm_jax.linear_gaussian_ssm.models import LinearGaussianSSM
from ssm_jax.linear_gaussian_ssm.inference import lgssm_posterior_sample


def lgssm_sample_demo(num_timesteps=100, key=jr.PRNGKey(0), sample_size=50):
    """Compare the function lgssm_posterior_sample to the counterpart in tfp 
    """
    # Setup the true model
    D_hid = 1
    D_obs = 1
    
    initial_mean = jnp.array([5.0])
    initial_covariance = jnp.eye(D_hid)
    dynamics_matrix = jnp.eye(D_hid) * 1.01
    dynamics_cov = jnp.eye(D_hid)
    emission_matrix = jnp.eye(D_obs)
    emission_cov = jnp.eye(D_obs) * 5.**2
    
    lgssm = LinearGaussianSSM(initial_mean=initial_mean, 
                              initial_covariance=initial_covariance,
                              dynamics_matrix=dynamics_matrix,
                              dynamics_covariance=dynamics_cov,
                              emission_matrix=emission_matrix,
                              emission_covariance=emission_cov)
    
    # Define the same model using tfd.LinearGaussianStateSpaceModel
    tfp_lgssm = lgssm_ssm_jax_to_tfp(num_timesteps, lgssm)
    
    # Generate true observation
    observations = tfp_lgssm.sample(seed=key)
    
    # Sample from the posterior distribution
    posterior_sample = partial(lgssm_posterior_sample, params=lgssm, emissions=observations)
    keys = jr.split(key, sample_size)
    _, samples = vmap(posterior_sample)(keys)
    
    tfp_samples = tfp_lgssm.posterior_sample(observations, seed=key, sample_shape=sample_size)
    
    # Plot the samples
    plt.xlabel("time")
    plt.xlim(0, num_timesteps - 1)
    plt.ylabel("posterior samples of states")
    
    plt.plot(observations, color='red')
    plt.plot(samples[:,:,0].T, alpha=0.12, color='blue')
    plt.plot(tfp_samples[:,:,0].T, alpha=0.12, color='green')
    plt.show()