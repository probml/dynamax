"""
Tests for the inference methods of the Linear Gaussian State Space Model.
"""
import pytest

import jax.scipy.linalg as jla
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from itertools import count
from functools import partial
from dynamax.linear_gaussian_ssm.inference import _get_params, _predict
from dynamax.linear_gaussian_ssm import (
    LinearGaussianSSM, ParamsLGSSM, ParamsLGSSMEmissions, lgssm_filter, lgssm_joint_sample,
)
from dynamax.utils.utils import has_tpu
from jax import tree, vmap
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

def _random_positive_definite_matrix(key, n):
    """Generate a matrix eligibly to use as a covariance matrix."""
    Q0 = jr.normal(key, shape=[n, n])
    Q_sym = (Q0 + Q0.T)/2
    I = jnp.eye(n)
    return Q_sym + n * I

def make_dynamic_lgssm_params(num_timesteps, latent_dim=2, observation_dim=4, seed=0):
    """Create a time-varying LGSSM with time-varying parameters."""
    key_seq = map(jr.key, count(seed))

    F = jr.normal(next(key_seq), shape=[num_timesteps - 1, latent_dim, latent_dim])
    keys = jr.split(next(key_seq), num=num_timesteps - 1)
    Q = vmap(partial(_random_positive_definite_matrix, n=latent_dim))(keys)

    H = jr.normal(next(key_seq), shape=[num_timesteps, observation_dim, latent_dim])
    keys = jr.split(next(key_seq), num=num_timesteps)
    R = vmap(partial(_random_positive_definite_matrix, n=observation_dim))(keys)
    b = jnp.zeros([num_timesteps - 1, latent_dim])
    d = jnp.zeros([num_timesteps, observation_dim])
    D = jnp.zeros([num_timesteps, observation_dim, 0])

    μ0 = jnp.zeros(latent_dim)
    Σ0 = jnp.eye(latent_dim)

    lgssm = LinearGaussianSSM(latent_dim, observation_dim)
    params, _ = lgssm.initialize(next(key_seq),
                                initial_mean=μ0,
                                initial_covariance=Σ0,
                                dynamics_weights=F,
                                dynamics_bias=b,
                                dynamics_covariance=Q,
                                emission_weights=H,
                                emission_bias=d,
                                emission_input_weights=D,
                                emission_covariance=R)
    return params, lgssm

class TestFilterMissingness:
    """
    Test filtering with partial and full missing emissions.
    """

    num_timesteps = 6
    key = jr.PRNGKey(1)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps, latent_dim=2, observation_dim=4)
    _, emissions = lgssm_joint_sample(params, key, num_timesteps)

    def _make_emission_covar_params_diagonal(self, params):
        emissions_covar = jnp.diagonal(params.emissions.cov, axis1=1, axis2=2)
        params_emissions = ParamsLGSSMEmissions(
            params.emissions.weights,
            params.emissions.bias,
            params.emissions.input_weights,
            emissions_covar,
        )
        return ParamsLGSSM(
            params.initial, params.dynamics, params_emissions,
        )

    @pytest.mark.parametrize("use_diagonal_emissions_covar", [True, False])
    def test_partial_missing_observations(self, use_diagonal_emissions_covar):
        """
        Test missing subvector of emissions.

        The following two cases should be equivalent.
        i) The same subvector of emissions is missing in all time points.
        ii) A measurement model corresponding to the (observed) subvector, with all
            emissions completely observed.
        """
        # Index 1 and 3 are missing. Represent by nan.
        is_observed = jnp.array([True, False, True, False])


        # Method i)
        y_partial_observed = jnp.where(is_observed[jnp.newaxis,:], self.emissions, jnp.nan)
        params = self.params
        if use_diagonal_emissions_covar:
            params = self._make_emission_covar_params_diagonal(self.params)
        posterior_method_i = lgssm_filter(params, y_partial_observed)

        # Method ii)
        y_subvector = self.emissions[:, is_observed]
        params_emissions = self.params.emissions
        sub_cov = params_emissions.cov[:, is_observed][...,is_observed]
        params_emissions_subvector = ParamsLGSSMEmissions(
            weights=params_emissions.weights[:, is_observed],
            bias=params_emissions.bias[:, is_observed],
            input_weights=params_emissions.input_weights[:, is_observed],
            cov=sub_cov,
        )
        params_subvector = ParamsLGSSM(
            initial=self.params.initial,
            dynamics=self.params.dynamics,
            emissions=params_emissions_subvector
        )
        if use_diagonal_emissions_covar:
            params_subvector = self._make_emission_covar_params_diagonal(params_subvector)
        posterior_method_ii = lgssm_filter(params_subvector, y_subvector)

        # Both methods must yield identical results.
        is_close = tree.map(allclose, posterior_method_i, posterior_method_ii)
        assert tree.all(is_close)

    @pytest.mark.parametrize("use_diagonal_emissions_covar", [True, False])
    def test_full_missing(self, use_diagonal_emissions_covar):
        """
        Test that a full missing emission skips the update step.

        The forwards filtering step consists of two steps:
        1) Predict the next state.
        2) Update the state using the observation.
        When an emission is completely missing, the Bayesian update corresponds to
        skipping 2).
        """
        t_missing = 2
        y_mid_missing = self.emissions.at[t_missing].set(jnp.nan)
        params = self.params
        if use_diagonal_emissions_covar:
            params = self._make_emission_covar_params_diagonal(self.params)
        posterior = lgssm_filter(params, y_mid_missing)

        not_na = tree.map(lambda x: jnp.all(~jnp.isnan(x)), posterior)
        assert tree.all(not_na)

        # Predict the next state.
        filtered_mean_tm1 = posterior.filtered_means[t_missing - 1]
        filtered_cov_tm1 = posterior.filtered_covariances[t_missing - 1]
        F, B, b, Q, *_ = _get_params(params, self.num_timesteps, t_missing - 1)
        u = jnp.zeros(B.shape[1:])
        pred_mean_t, pred_cov_t = _predict(filtered_mean_tm1, filtered_cov_tm1, F, B, b, Q, u)

        # Check that the filtered mean is corresponds to skipping the update step.
        filtered_mean_t = posterior.filtered_means[t_missing]
        filtered_cov_t = posterior.filtered_covariances[t_missing]
        assert allclose(filtered_mean_t, pred_mean_t)
        assert allclose(filtered_cov_t, pred_cov_t)

    def test_log_likelihood_last_missing(self):
        """Test the log-likelihood with a missing emission.

        When emission y[t] is completely missing, p(y[1:t]) = p(y[1:t-1]).
        """
        def _trim_params(params):
            return ParamsLGSSM(
                params.initial,
                tree.map(lambda x: x[:-1], params.dynamics),
                tree.map(lambda x: x[:-1], params.emissions),
            )

        # Method i):
        # Compute the log-likelihood of a fully observed sequence not including the last
        # emission.
        t_missing = -1
        params_Tmin1 = _trim_params(self.params)
        posterior_Tmin1 = lgssm_filter(params_Tmin1, self.emissions[:-1])

        # Method ii):
        # Compute the log-likelihood of a sequence with the last emission missing.
        y_last_missing = self.emissions.at[t_missing].set(jnp.nan)
        posterior_last_missing = lgssm_filter(self.params, y_last_missing)

        allclose(posterior_last_missing.marginal_loglik, posterior_Tmin1.marginal_loglik)
