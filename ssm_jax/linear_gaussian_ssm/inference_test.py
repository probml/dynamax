from jax import random as jr
from jax import numpy as jnp
from jax import vmap

import tensorflow_probability.substrates.jax.distributions as tfd
from ssm_jax.linear_gaussian_ssm.models.linear_gaussian_ssm import LinearGaussianSSM


def lgssm_ssm_jax_to_tfp(num_timesteps, params):
    """Create a Tensorflow Probability `LinearGaussianStateSpaceModel` object
     from an ssm_jax `LinearGaussianSSM`.

    Args:
        num_timesteps: int, the number of timesteps.
        lgssm: LinearGaussianSSM or LGSSMParams object.
    """
    dynamics_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=params['dynamics']['cov'])
    emission_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=params['emissions']['cov'])
    initial_dist = tfd.MultivariateNormalFullCovariance(params['initial']['mean'], params['initial']['cov'])

    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        params['dynamics']['weights'],
        dynamics_noise_dist,
        params['emissions']['weights'],
        emission_noise_dist,
        initial_dist,
    )

    return tfp_lgssm


def test_kalman_filter(num_timesteps=5, seed=0):
    key = jr.PRNGKey(seed)
    init_key, sample_key = jr.split(key)

    state_dim = 4
    emission_dim = 2
    delta = 1.0

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.random_initialization(init_key)
    params['initial']['mean'] = jnp.array([8.0, 10.0, 1.0, 0.0])
    params['initial']['cov'] = jnp.eye(state_dim) * 0.1
    params['dynamics']['weights'] = jnp.array([[1, 0, delta, 0],
                                               [0, 1, 0, delta],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]])
    params['dynamics']['cov'] = jnp.eye(state_dim) * 0.001
    params['emissions']['weights'] = jnp.array([[1.0, 0, 0, 0],
                                                [0, 1.0, 0, 0]])
    params['emissions']['cov'] = jnp.eye(emission_dim) * 1.0

    # Sample data and compute posterior
    _, emissions = lgssm.sample(params, sample_key, num_timesteps)
    ssm_posterior = lgssm.smoother(params, emissions)

    # TensorFlow Probability posteriors
    tfp_lgssm = lgssm_ssm_jax_to_tfp(num_timesteps, params)
    tfp_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(emissions)
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(emissions)

    assert jnp.allclose(ssm_posterior.filtered_means, tfp_filtered_means, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.filtered_covariances, tfp_filtered_covs, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.smoothed_means, tfp_smoothed_means, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.smoothed_covariances, tfp_smoothed_covs, rtol=1e-2)
    assert jnp.allclose(ssm_posterior.marginal_loglik, tfp_lls.sum())
