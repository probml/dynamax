from functools import partial

from jax import random as jr
from jax import numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt

import tensorflow_probability.substrates.jax.distributions as tfd

from ssm_jax.lgssm.inference import lgssm_filter, lgssm_posterior_sample
from ssm_jax.lgssm.models import LinearGaussianSSM


def lgssm_ssm_jax_to_tfp(num_timesteps, lgssm):
    """Create a Tensorflow Probability `LinearGaussianStateSpaceModel` object
     from an ssm_jax `LinearGaussianSSM`.

    Args:
        num_timesteps: int, the number of timesteps.
        lgssm: LinearGaussianSSM or LGSSMParams object.
    """
    dynamics_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=lgssm.dynamics_covariance)
    emission_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=lgssm.emission_covariance)
    initial_dist = tfd.MultivariateNormalFullCovariance(lgssm.initial_mean, lgssm.initial_covariance)

    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        lgssm.dynamics_matrix,
        dynamics_noise_dist,
        lgssm.emission_matrix,
        emission_noise_dist,
        initial_dist,
    )

    return tfp_lgssm


def test_kalman_filter(num_timesteps=5, seed=0):

    delta = 1.0
    F = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])

    H = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    state_size, _ = F.shape
    observation_size, _ = H.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 1.0

    # Prior parameter distribution
    mu0 = jnp.array([8.0, 10.0, 1.0, 0.0])
    Sigma0 = jnp.eye(state_size) * 0.1

    lgssm = LinearGaussianSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=H,
        emission_covariance=R,
    )

    tfp_lgssm = lgssm_ssm_jax_to_tfp(num_timesteps, lgssm)

    ### Sample data ###
    key = jr.PRNGKey(seed)
    states, emissions = lgssm.sample(key, num_timesteps)

    # ssm_jax posteriors
    ssm_posterior = lgssm.smoother(emissions)

    # TensorFlow Probability posteriors
    tfp_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(emissions)
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(emissions)

    assert jnp.allclose(ssm_posterior.filtered_means, tfp_filtered_means, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.filtered_covariances, tfp_filtered_covs, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.smoothed_means, tfp_smoothed_means, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.smoothed_covariances, tfp_smoothed_covs, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.marginal_loglik, tfp_lls.sum())

