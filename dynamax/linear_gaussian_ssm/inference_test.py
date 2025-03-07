"""
Tests for the inference methods of the Linear Gaussian State Space Model.
"""
import pytest

import jax.scipy.linalg as jla
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from functools import partial
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.utils.utils import has_tpu
from jax import vmap
from jax import random as jr

# Use different tolerance threshold for TPU
if has_tpu():
    allclose = partial(jnp.allclose, atol=1e-1)
else:
    allclose = partial(jnp.allclose, atol=1e-4)


def flatten_diagonal_emission_cov(params):
    """Flatten a diagonal emission covariance matrix into a vector.

    Args:
        params: LGSSMParams object.

    Returns:
        params: LGSSMParams object with flattened diagonal emission covariance.
    """
    R = params.emissions.cov

    if R.ndim == 2:
        R_diag = jnp.diag(R)
        R_full = jnp.diag(R_diag)
    else:
        R_diag = vmap(jnp.diag)(R)
        R_full = vmap(jnp.diag)(R_diag)

    assert allclose(R, R_full), "R is not diagonal"

    emission_params_diag = params.emissions._replace(cov=R_diag)
    params = params._replace(emissions=emission_params_diag)
    return params


def joint_posterior_mvn(params, emissions):
    """Construct the joint posterior MVN of a LGSSM, by inverting the joint precision matrix which
    has a known block tridiagonal form.

    Args:
        params: LGSSMParams object.
        emissions: Emission data.

    Returns: 
        means: jnp.ndarray, shape (num_timesteps, state_dim), the joint posterior means.
        Sigma_diag_blocks: jnp.ndarray, shape (num_timesteps, state_dim, state_dim), the joint posterior covariance diagonal blocks.
    """
    Q = params.dynamics.cov
    R = params.emissions.cov
    F = params.dynamics.weights
    H = params.emissions.weights
    mu0 = params.initial.mean
    Sigma0 = params.initial.cov
    num_timesteps = emissions.shape[0]
    state_dim = params.dynamics.weights.shape[0]
    emission_dim = params.emissions.weights.shape[0]
    Qinv = jnp.linalg.inv(Q)
    Rinv = jnp.linalg.inv(R)
    Sigma0inv = jnp.linalg.inv(Sigma0)

    # Construct the big precision matrix (block tridiagonal)
    # set up small blocks
    Omega1 = F.T @ Qinv @ F + H.T @ Rinv @ H + Sigma0inv
    Omegat = F.T @ Qinv @ F + H.T @ Rinv @ H + Qinv
    OmegaT = Qinv + H.T @ Rinv @ H
    OmegaC = - F.T @ Qinv

    # construct big block diagonal matrix
    blocks = [Omega1] + [Omegat] * (num_timesteps-2) + [OmegaT]
    Omega_diag = jla.block_diag(*blocks)

    # construct big block super/sub-diagonal matrices and sum
    aux = jnp.empty((0, state_dim), int)
    blocks = [OmegaC] * (num_timesteps-1)
    Omega_superdiag = jla.block_diag(aux, *blocks, aux.T)
    Omega_subdiag = Omega_superdiag.T
    Omega = Omega_diag + Omega_superdiag + Omega_subdiag

    # Compute the joint covariance matrix
    # diagonal blocks are the smoothed covariances (marginals of the full joint)
    Sigma = jnp.linalg.inv(Omega)
    covs = jnp.array([Sigma[i:i+state_dim, i:i+state_dim] for i in range(0, num_timesteps*state_dim, state_dim)])

    # Compute the means (these are the smoothing means)
    # they are the solution to the big linear system Omega @ means = rhs
    padded = jnp.pad(Sigma0inv @ mu0, (0, (num_timesteps-1)*state_dim ), constant_values=0).reshape(num_timesteps * state_dim, 1)
    rhs = jla.block_diag(*[H.T @ Rinv] * num_timesteps) @ emissions.reshape((num_timesteps*emission_dim, 1)) + padded
    means = Sigma @ rhs
    means = means.reshape((num_timesteps, state_dim))

    return means, covs


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


class TestFilteringAndSmoothing:
    """
    Tests for the filtering and smoothing methods of the Linear Gaussian State Space
    """
    key = jr.PRNGKey(0)
    num_timesteps = 15
    num_samples = 1000
    state_dim = 4
    emission_dim = 2

    k1, k2, k3 = jr.split(key, 3)
    
    # Construct an LGSSM with simple dynamics and emissions
    mu0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    Sigma0 = jnp.eye(state_dim) * 0.1
    F = jnp.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=jnp.float32)
    Q = jnp.eye(state_dim) * 0.001
    H = jnp.array([[1.0, 0, 0, 0],
                     [0, 1.0, 0, 0]])
    R = jnp.eye(emission_dim) * 1.0

    lgssm = LinearGaussianSSM(state_dim, emission_dim)
    params, _ = lgssm.initialize(k1,
                                 initial_mean=mu0,
                                 initial_covariance=Sigma0,
                                 dynamics_weights=F,
                                 dynamics_covariance=Q,
                                 emission_weights=H,
                                 emission_covariance=R)

    # Sample random emissions
    _, emissions = lgssm.sample(params, k2, num_timesteps)

    # Run the smoother with the full covariance parameterization
    posterior = lgssm.smoother(params, emissions)

    # Run the smoother with the diagonal covariance parameterization
    params_diag = flatten_diagonal_emission_cov(params)
    ssm_posterior_diag = lgssm.smoother(params_diag, emissions)

    # Sample from the posterior distribution
    posterior_sample = partial(lgssm.posterior_sample, 
                                params=params, 
                                emissions=emissions)
    samples = vmap(posterior_sample)(jr.split(k3, num_samples))
        
    def test_smoother_vs_tfp(self):
        """Test that the dynamax and TFP implementations of the Kalman filter are consistent."""
        tfp_lgssm = lgssm_dynamax_to_tfp(self.num_timesteps, self.params)
        tfp_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(self.emissions)
        tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(self.emissions)

        assert allclose(self.posterior.filtered_means, tfp_filtered_means)
        assert allclose(self.posterior.filtered_covariances, tfp_filtered_covs)
        assert allclose(self.posterior.smoothed_means, tfp_smoothed_means)
        assert allclose(self.posterior.smoothed_covariances, tfp_smoothed_covs)
        assert allclose(self.posterior.marginal_loglik, tfp_lls.sum())

        # Compare posterior with diagonal emission covariance
        assert allclose(self.ssm_posterior_diag.filtered_means, tfp_filtered_means)
        assert allclose(self.ssm_posterior_diag.filtered_covariances, tfp_filtered_covs)
        assert allclose(self.ssm_posterior_diag.smoothed_means, tfp_smoothed_means)
        assert allclose(self.ssm_posterior_diag.smoothed_covariances, tfp_smoothed_covs)
        assert allclose(self.ssm_posterior_diag.marginal_loglik, tfp_lls.sum())
        

    def test_kalman_vs_joint(self):
        """Test that the dynamax and joint posterior methods are consistent."""
        joint_means, joint_covs = joint_posterior_mvn(self.params, self.emissions)

        assert allclose(self.posterior.smoothed_means, joint_means)
        assert allclose(self.posterior.smoothed_covariances, joint_covs)
        assert allclose(self.ssm_posterior_diag.smoothed_means, joint_means)
        assert allclose(self.ssm_posterior_diag.smoothed_covariances, joint_covs)

    
    def test_posterior_samples(self):
        """Test that posterior samples match the mean of the smoother"""
        monte_carlo_var = vmap(jnp.diag)(self.posterior.smoothed_covariances) / self.num_samples
        assert jnp.all(abs(jnp.mean(self.samples, axis=0) - self.posterior.smoothed_means) < 6 * jnp.sqrt(monte_carlo_var))
