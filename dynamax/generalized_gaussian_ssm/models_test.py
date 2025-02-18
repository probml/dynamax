"""
Tests for the GeneralizedGaussianSSM class.
"""
import pytest
import jax.random as jr
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from dynamax.generalized_gaussian_ssm import GeneralizedGaussianSSM, ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, EKFIntegrals

NUM_TIMESTEPS = 100

CONFIGS = [
    (jr.PRNGKey(0), dict(state_dim=3, emission_dim=5)),
    (jr.PRNGKey(1), dict(state_dim=5, emission_dim=5)),
    (jr.PRNGKey(2), dict(state_dim=10, emission_dim=7)),
]

@pytest.mark.parametrize(["key", "kwargs"], CONFIGS)
def test_poisson_emission(key, kwargs):
    """
    Test that the marginal log-likelihood under Poisson emission is higher than 
    when we treat the count emissions as Gaussian random variables.
    """
    keys = jr.split(key, 3)
    state_dim = kwargs['state_dim']
    emission_dim = 1 # Univariate Poisson
    poisson_weights = jr.normal(keys[0], shape=(emission_dim, state_dim)) /  jnp.sqrt(state_dim)
    model = GeneralizedGaussianSSM(state_dim, emission_dim)
    
    # Define model parameters with Poisson emission
    pois_params = ParamsGGSSM(
        initial_mean=jr.normal(keys[1], (state_dim,)),
        initial_covariance=jnp.eye(state_dim),
        dynamics_function=lambda z: 0.99 * z,
        dynamics_covariance=0.001*jnp.eye(state_dim),
        emission_mean_function=lambda z: jnp.exp(poisson_weights @ z),
        emission_cov_function = lambda z: jnp.exp(poisson_weights @ z),
        emission_dist=lambda mu, Sigma: tfd.Poisson(log_rate = jnp.log(mu))
    )
    _, emissions = model.sample(pois_params, keys[2], num_timesteps=NUM_TIMESTEPS)

    # Define model parameters with default Gaussian emission
    gaussian_params = ParamsGGSSM(
        initial_mean=jr.normal(keys[1], (state_dim,)),
        initial_covariance=jnp.eye(state_dim),
        dynamics_function=lambda z: 0.99 * z,
        dynamics_covariance=0.001*jnp.eye(state_dim),
        emission_mean_function=lambda z: jnp.exp(poisson_weights @ z),
        emission_cov_function=lambda z: jnp.exp(poisson_weights @ z)
    )

    # Fit model with Poisson emission
    pois_marginal_lls = conditional_moments_gaussian_filter(pois_params, EKFIntegrals(), emissions).marginal_loglik

    # Fit model with Gaussian emission
    gaussian_marginal_lls = conditional_moments_gaussian_filter(gaussian_params, EKFIntegrals(), emissions).marginal_loglik
    
    # Check that the marginal log-likelihoods under Poisson emission are higher
    assert pois_marginal_lls > gaussian_marginal_lls
