from functools import partial

from jax import vmap
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

from ssm_jax.linear_gaussian_ssm.inference_test import lgssm_ssm_jax_to_tfp
from ssm_jax.linear_gaussian_ssm.models import LinearGaussianSSM


def lgssm_sample_demo(num_timesteps=100, key=jr.PRNGKey(0), sample_size=50, test_mode=False):
    """Compare the function lgssm_posterior_sample to the counterpart in tfp
    """
    # Setup the true model
    state_dim = 1
    emission_dim = 1

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.random_initialization(key)
    params['initial']['mean'] = jnp.array([5.0])
    params['initial']['cov'] = jnp.eye(state_dim)
    params['dynamics']['weights'] = jnp.eye(state_dim) * 1.01
    params['dynamics']['cov'] = jnp.eye(state_dim)
    params['emissions']['weights'] = jnp.eye(emission_dim)
    params['emissions']['cov'] = jnp.eye(emission_dim) * 5.**2

    # Generate true observation
    sample_key, key = jr.split(key)
    states, emissions = lgssm.sample(params, key=sample_key, num_timesteps=num_timesteps)

    # Sample from the posterior distribution
    posterior_sample = partial(lgssm.posterior_sample, params=params, emissions=emissions)
    keys = jr.split(key, sample_size)
    samples = vmap(lambda key: posterior_sample(key=key))(keys)

    # Do the same with TFP
    tfp_lgssm = lgssm_ssm_jax_to_tfp(num_timesteps, params)
    tfp_samples = tfp_lgssm.posterior_sample(emissions, seed=key, sample_shape=sample_size)

    if not test_mode:
        # Plot the samples
        plt.xlabel("time")
        plt.xlim(0, num_timesteps - 1)
        plt.ylabel("posterior samples of states")

        plt.plot(emissions, color='red')
        plt.plot(samples[:,:,0].T, alpha=0.12, color='blue')
        plt.plot(tfp_samples[:,:,0].T, alpha=0.12, color='green')
        plt.show()


if __name__ == "__main__":
    lgssm_sample_demo()
