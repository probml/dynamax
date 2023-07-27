from jax import vmap
from jax import random as jr
import jax.scipy.linalg as jla
import jax.numpy as jnp
from functools import partial

import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

from dynamax.utils.utils import has_tpu

if has_tpu():
    def allclose(x, y, atol=1e-1):
        return jnp.allclose(x, y, atol=atol)
else:
    def allclose(x, y, atol=1e-1):
        return jnp.allclose(x, y, atol=atol)

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

def build_lgssm_for_inference():
    """Construct example LinearGaussianSSM object for testing.
    """

    key = jr.PRNGKey(0)
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

    return lgssm, params

def build_lgssm_for_sampling():
    state_dim = 1
    emission_dim = 1
    key = jr.PRNGKey(0)
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
    return lgssm, params

class TestFilteringAndSmoothing():

    ## For inference tests
    lgssm, params = build_lgssm_for_inference()

     # Sample data and compute dynamax posteriors
    sample_key = jr.PRNGKey(0)
    num_timesteps = 15
    _, emissions = lgssm.sample(params, sample_key, num_timesteps)
    ssm_posterior = lgssm.filter(params, emissions)
    print(ssm_posterior.filtered_means.shape)

    ssm_posterior = lgssm.smoother(params, emissions)
    print(ssm_posterior.filtered_means.shape)
    print(ssm_posterior.smoothed_means.shape)

    # repeat sampling with NaNs in the emissions
    nan_x = (0, emissions.shape[0], 0, emissions.shape[0])
    nan_y = (0, emissions.shape[0], emissions.shape[1], emissions.shape[1])
    emissions_nan = emissions.at[nan_x, nan_y].set(jnp.nan)
    ssm_posterior_nan = lgssm.smoother(params, emissions_nan)

    # TensorFlow Probability posteriors
    tfp_lgssm = lgssm_dynamax_to_tfp(num_timesteps, params)
    tfp_lls, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(emissions)
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(emissions)

    # Posteriors from full joint distribution
    joint_means, joint_covs = joint_posterior_mvn(params, emissions)


    ## For sampling tests
    lgssm, params = build_lgssm_for_sampling()

    # Generate true observation
    num_timesteps=100
    sample_size=500
    key = jr.PRNGKey(0)
    sample_key, key = jr.split(key)
    states, emissions = lgssm.sample(params, key=sample_key, num_timesteps=num_timesteps)

    # Sample from the posterior distribution
    posterior_sample = partial(lgssm.posterior_sample, params=params, emissions=emissions)
    keys = jr.split(key, sample_size)
    samples = vmap(lambda key, func=posterior_sample: func(key=key))(keys)

    # Do the same with TFP
    tfp_lgssm = lgssm_dynamax_to_tfp(num_timesteps, params)
    tfp_samples = tfp_lgssm.posterior_sample(emissions, seed=key, sample_shape=sample_size) 

    def test_kalman_tfp(self):
        assert allclose(self.ssm_posterior.filtered_means, self.tfp_filtered_means)
        assert allclose(self.ssm_posterior.filtered_covariances, self.tfp_filtered_covs)
        assert allclose(self.ssm_posterior.smoothed_means, self.tfp_smoothed_means)
        assert allclose(self.ssm_posterior.smoothed_covariances, self.tfp_smoothed_covs)
        assert allclose(self.ssm_posterior.marginal_loglik, self.tfp_lls.sum())

    def test_kalman_tfp_nan(self):
        assert allclose(self.ssm_posterior_nan.filtered_means, self.tfp_filtered_means, atol=1e0)
        assert allclose(self.ssm_posterior_nan.filtered_covariances, self.tfp_filtered_covs, atol=1e0)
        assert allclose(self.ssm_posterior_nan.smoothed_means, self.tfp_smoothed_means, atol=1e0)
        assert allclose(self.ssm_posterior_nan.smoothed_covariances, self.tfp_smoothed_covs, atol=1e0)

    def test_kalman_vs_joint(self):
        assert allclose(self.ssm_posterior.smoothed_means, self.joint_means)
        assert allclose(self.ssm_posterior.smoothed_covariances, self.joint_covs)

    def test_posterior_sampler(self):
        assert allclose(jnp.mean(self.samples), jnp.mean(self.tfp_samples))
        assert allclose(jnp.std(self.samples), jnp.std(self.tfp_samples))
        