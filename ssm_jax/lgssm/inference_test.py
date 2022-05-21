from jax import random as jr
from jax import numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfd

from ssm_jax.lgssm.inference import lgssm_filter
from ssm_jax.lgssm.models import LinearGaussianSSM

def tfp_filter(dynamics_matrix,
               dynamics_noise_scale,
               emission_matrix,
               emission_noise_scale,
               initial_mean,
               emissions):
    """ Perform filtering using tensorflow probability """
    num_timesteps = len(emissions)
    state_dim, _ = dynamics_matrix.shape
    emission_dim, _ = emission_matrix.shape

    # Make the initial and noise distributions
    dynamics_noise_dist = tfd.MultivariateNormalDiag(
        scale_diag=jnp.ones(state_dim) * dynamics_noise_scale)
    emission_noise_dist = tfd.MultivariateNormalDiag(
        scale_diag=jnp.ones(emission_dim) * emission_noise_scale)
    initial_dist = tfd.MultivariateNormalDiag(initial_mean, jnp.ones([state_dim]))

    LGSSM = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        dynamics_matrix, dynamics_noise_dist,
        emission_matrix, emission_noise_dist,
        initial_dist)

    _, filtered_means, filtered_covs, _, _, _, _ = LGSSM.forward_filter(emissions)
    return filtered_means, filtered_covs


def test_kalman_filter(state_dim=2, emission_dim=2,
                       dynamics_noise_scale=1.0,
                       emission_noise_scale=1.0,
                       num_timesteps=15,
                       seed=0):
    lgssm = LinearGaussianSSM(
        initial_mean=jnp.array([8, 10]).astype(float),
        initial_covariance=jnp.eye(state_dim) * 1.0,
        dynamics_matrix=jnp.eye(state_dim),
        dynamics_bias=jnp.zeros((state_dim,)),
        dynamics_covariance=jnp.eye(state_dim) * dynamics_noise_scale,
        emission_matrix=jnp.eye(state_dim),  # assumes emission_dim == state_dim
        emission_bias=jnp.zeros((emission_dim,)),
        emission_covariance=jnp.eye(emission_dim) * emission_noise_scale)

    ### Sample data ###
    key = jr.PRNGKey(seed)
    states, emissions = lgssm.sample(key, num_timesteps)

    ssm_ll_filt, ssm_filtered_means, ssm_filtered_covs = lgssm.filter(emissions)
    tfp_filtered_means, tfp_filtered_covs = tfp_filter(
        lgssm.dynamics_matrix,
        dynamics_noise_scale,
        lgssm.emission_matrix,
        emission_noise_scale,
        lgssm.initial_mean,
        emissions)

    assert jnp.allclose(ssm_filtered_means, tfp_filtered_means, rtol=1e-2)
    assert jnp.allclose(ssm_filtered_covs, tfp_filtered_covs, rtol=1e-2)
