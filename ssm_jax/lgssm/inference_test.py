from jax import random as jr
from jax import numpy as jnp

import tensorflow_probability.substrates.jax.distributions as tfd

from ssm_jax.lgssm.inference import lgssm_filter
from ssm_jax.lgssm.models import LinearGaussianSSM

def lgssm_ssm_jax_to_tfp(num_timesteps, lgssm):
    """ Create a Tensorflow Probability `LinearGaussianStateSpaceModel` object
     from an ssm_jax `LinearGaussianSSM`.

    Args:
        num_timesteps: int, the number of timesteps.
        lgssm: LinearGaussianSSM or LGSSMParams object.
    """
    dynamics_noise_dist = tfd.MultivariateNormalFullCovariance(
            covariance_matrix=lgssm.dynamics_covariance)
    emission_noise_dist = tfd.MultivariateNormalFullCovariance(
            covariance_matrix=lgssm.emission_covariance)
    initial_dist = tfd.MultivariateNormalFullCovariance(
            lgssm.initial_mean, lgssm.initial_covariance)

    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        lgssm.dynamics_matrix, dynamics_noise_dist,
        lgssm.emission_matrix, emission_noise_dist,
        initial_dist)
    
    return tfp_lgssm


def test_kalman_filter(state_dim=2, emission_dim=2,
                       dynamics_noise_scale=0.5,
                       emission_noise_scale=0.5,
                       num_timesteps=15,
                       seed=0):

    lgssm = LinearGaussianSSM(
        initial_mean=jnp.array([8, 10]).astype(float),
        initial_covariance=jnp.eye(state_dim) * 1.0,
        dynamics_matrix=jnp.eye(state_dim),
        dynamics_covariance=jnp.eye(state_dim) * dynamics_noise_scale ** 2,
        emission_matrix=jnp.eye(state_dim),  # assumes emission_dim == state_dim
        emission_covariance=jnp.eye(emission_dim) * emission_noise_scale ** 2)

    tfp_lgssm = lgssm_ssm_jax_to_tfp(num_timesteps, lgssm)

    ### Sample data ###
    key = jr.PRNGKey(seed)
    states, emissions = lgssm.sample(key, num_timesteps)

    ssm_ll_filt, ssm_filtered_means, ssm_filtered_covs = lgssm.filter(emissions)
    ssm_posterior = lgssm.smoother(emissions)
    ssm_ll_post = ssm_posterior.marginal_log_lkhd

    tfp_filtered_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(emissions)
    tfp_ll_filt = tfp_filtered_lls.sum()
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(emissions)
    tfp_ll_post = tfp_lgssm.log_prob(emissions)

    assert jnp.allclose(ssm_filtered_means, tfp_filtered_means, rtol=1e-2)
    assert jnp.allclose(ssm_filtered_covs, tfp_filtered_covs, rtol=1e-2)
    assert jnp.allclose(ssm_ll_filt, tfp_ll_filt)
    assert jnp.allclose(ssm_posterior.smoothed_means, tfp_smoothed_means, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.smoothed_covariances, tfp_smoothed_covs, rtol=1e-2)
    assert jnp.allclose(ssm_ll_post, tfp_ll_post)
