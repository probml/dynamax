from jax import vmap
from jax import random as jr
import jax.numpy as jnp
from functools import partial

import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

from dynamax.utils.utils import has_tpu

if has_tpu():
    def allclose(x, y):
        return jnp.allclose(x, y, atol=1e-1)
else:
    def allclose(x,y):
        return jnp.allclose(x, y, atol=1e-1)

def lgssm_dynamax_to_tfp(num_timesteps, params):
    """Create a Tensorflow Probability `LinearGaussianStateSpaceModel` object
     from an dynamax `LinearGaussianSSM`.

    Args:
        num_timesteps: int, the number of timesteps.
        lgssm: LinearGaussianSSM or LGSSMParams object.
    """
    dynamics_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=params.dynamics.cov)
    emission_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=params.emissions.cov)
    initial_dist = tfd.MultivariateNormalFullCovariance(params.initial.mean, params.initial.cov)

    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        params.dynamics.weights,
        dynamics_noise_dist,
        params.emissions.weights,
        emission_noise_dist,
        initial_dist,
    )

    return tfp_lgssm


def test_kalman(num_timesteps=5, seed=0):
    key = jr.PRNGKey(seed)
    init_key, sample_key = jr.split(key)

    state_dim = 4
    emission_dim = 2
    delta = 1.0

    mu0 = jnp.array([8.0, 10.0, 1.0, 0.0])
    Sigma0 = jnp.eye(state_dim) * 0.1
    F = jnp.array([[1, 0, delta, 0],
                    [0, 1, 0, delta],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    Q = jnp.eye(state_dim) * 0.001
    H = jnp.array([[1.0, 0, 0, 0],
                     [0, 1.0, 0, 0]])
    R = jnp.eye(emission_dim) * 1.0

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.initialize(key,
                                 initial_mean=mu0,
                                 initial_covariance=Sigma0,
                                 dynamics_weights=F,
                                 dynamics_covariance=Q,
                                 emission_weights=H,
                                 emission_covariance=R)


    # Sample data and compute posterior
    _, emissions = lgssm.sample(params, sample_key, num_timesteps)
    ssm_posterior = lgssm.filter(params, emissions)
    print(ssm_posterior.filtered_means.shape)

    ssm_posterior = lgssm.smoother(params, emissions)
    print(ssm_posterior.filtered_means.shape)
    print(ssm_posterior.smoothed_means.shape)

    # TensorFlow Probability posteriors
    tfp_lgssm = lgssm_dynamax_to_tfp(num_timesteps, params)
    tfp_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(emissions)
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(emissions)

    assert allclose(ssm_posterior.filtered_means, tfp_filtered_means)
    assert allclose(ssm_posterior.filtered_covariances, tfp_filtered_covs)
    assert allclose(ssm_posterior.smoothed_means, tfp_smoothed_means)
    assert allclose(ssm_posterior.smoothed_covariances, tfp_smoothed_covs)
    assert allclose(ssm_posterior.marginal_loglik, tfp_lls.sum())


def test_posterior_sampler():
    state_dim = 1
    emission_dim = 1

    num_timesteps=100
    key = jr.PRNGKey(0)
    sample_size=500

    mu0 = jnp.array([5.0])
    Sigma0 = jnp.eye(state_dim)
    F = jnp.eye(state_dim) * 1.01
    Q = jnp.eye(state_dim)
    H = jnp.eye(emission_dim)
    R = jnp.eye(emission_dim) * 5.**2

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.initialize(key,
                                 initial_mean=mu0,
                                 initial_covariance=Sigma0,
                                 dynamics_weights=F,
                                 dynamics_covariance=Q,
                                 emission_weights=H,
                                 emission_covariance=R)

    # Generate true observation
    sample_key, key = jr.split(key)
    states, emissions = lgssm.sample(params, key=sample_key, num_timesteps=num_timesteps)

    # Sample from the posterior distribution
    posterior_sample = partial(lgssm.posterior_sample, params=params, emissions=emissions)
    keys = jr.split(key, sample_size)
    samples = vmap(lambda key: posterior_sample(key=key))(keys)

    # Do the same with TFP
    tfp_lgssm = lgssm_dynamax_to_tfp(num_timesteps, params)
    tfp_samples = tfp_lgssm.posterior_sample(emissions, seed=key, sample_shape=sample_size)

    print(samples.shape) # (N,T,1)
    print(tfp_samples.shape) # (N,T,1)

    assert allclose(jnp.mean(samples), jnp.mean(tfp_samples))
    assert allclose(jnp.std(samples), jnp.std(tfp_samples))
