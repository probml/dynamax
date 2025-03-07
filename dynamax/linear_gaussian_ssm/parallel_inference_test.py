"""
Tests for parallel inference in linear Gaussian SSMs.
"""
import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_joint_sample
from dynamax.linear_gaussian_ssm import lgssm_smoother as serial_lgssm_smoother, lgssm_filter as serial_lgssm_filter
from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother, parallel_lgssm_filter
from dynamax.linear_gaussian_ssm import lgssm_posterior_sample as serial_lgssm_posterior_sample
from dynamax.linear_gaussian_ssm import parallel_lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference_test import flatten_diagonal_emission_cov
from functools import partial
from jax import vmap


allclose = partial(jnp.allclose, atol=1e-4)    

def make_static_lgssm_params():
    """Create a static LGSSM with fixed parameters."""
    dt = 0.1
    F = jnp.eye(4) + dt * jnp.eye(4, k=2)
    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                          [dt**2/2, dt]]),
                         jnp.eye(2))
                         
    H = jnp.eye(2, 4)
    R = 0.5 ** 2 * jnp.eye(2)
    μ0 = jnp.array([0.,0.,1.,-1.])
    Σ0 = jnp.eye(4)

    latent_dim = 4
    observation_dim = 2

    lgssm = LinearGaussianSSM(latent_dim, observation_dim)
    params, _ = lgssm.initialize(jr.PRNGKey(0),
                             initial_mean=μ0,
                             initial_covariance= Σ0,
                             dynamics_weights=F,
                             dynamics_covariance=Q,
                             emission_weights=H,
                             emission_covariance=R)
    return params, lgssm


def make_lgssm_params_with_inputs():
    """Create a static LGSSM with fixed parameters and inputs."""
    dt = 0.1
    F = jnp.eye(4) + dt * jnp.eye(4, k=2)
    B = jnp.array([[0., 0.], [1., 0.], [0., 0.], [0., 1.]]) * dt
    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                          [dt**2/2, dt]]),
                         jnp.eye(2))
                         
    H = jnp.eye(2, 4)
    D = jnp.ones((2, 2))
    R = 0.5 ** 2 * jnp.eye(2)
    μ0 = jnp.array([0.,0.,1.,-1.])
    Σ0 = jnp.eye(4)

    latent_dim = 4
    observation_dim = 2
    input_dim = 2

    lgssm = LinearGaussianSSM(latent_dim, observation_dim, input_dim)
    params, _ = lgssm.initialize(jr.PRNGKey(0),
                             initial_mean=μ0,
                             initial_covariance= Σ0,
                             dynamics_weights=F,
                             dynamics_input_weights=B,
                             dynamics_covariance=Q,
                             emission_weights=H,
                             emission_input_weights=D,
                             emission_covariance=R)
    return params, lgssm
        

def make_dynamic_lgssm_params(num_timesteps, latent_dim=4, observation_dim=2, seed=0):
    """Create a time-varying LGSSM with time-varying parameters."""
    key = jr.PRNGKey(seed)
    key, key_f, key_r, key_init = jr.split(key, 4)

    dt = 0.1
    f_scale = jr.normal(key_f, (num_timesteps,)) * 0.5
    F = f_scale[:,None,None] * jnp.tile(jnp.eye(latent_dim), (num_timesteps, 1, 1))
    F += dt * jnp.eye(latent_dim, k=2)

    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                                 [dt**2/2, dt]]),
                      jnp.eye(latent_dim // 2))
    assert Q.shape[-1] == latent_dim
    H = jnp.eye(observation_dim, latent_dim)

    r_scale = jr.normal(key_r, (num_timesteps,)) * 0.1
    R = (r_scale**2)[:,None,None] * jnp.tile(jnp.eye(observation_dim), (num_timesteps, 1, 1))
    
    μ0 = jnp.array([0.,0.,1.,-1.])
    Σ0 = jnp.eye(latent_dim)

    lgssm = LinearGaussianSSM(latent_dim, observation_dim)
    params, _ = lgssm.initialize(key_init,
                                initial_mean=μ0,
                                initial_covariance=Σ0,
                                dynamics_weights=F,
                                dynamics_covariance=Q,
                                emission_weights=H,
                                emission_covariance=R)
    return params, lgssm


class TestParallelLGSSMSmoother:
    """ Compare parallel and serial lgssm smoothing implementations."""
    
    num_timesteps = 50
    key = jr.PRNGKey(1)

    params, lgssm = make_static_lgssm_params()   
    params_diag = flatten_diagonal_emission_cov(params)
    _, emissions = lgssm_joint_sample(params, key, num_timesteps)

    serial_posterior = serial_lgssm_smoother(params, emissions)
    parallel_posterior = parallel_lgssm_smoother(params, emissions)
    parallel_posterior_diag = parallel_lgssm_smoother(params_diag, emissions)

    def test_filtered_means(self):
        """Check if filtered means are close."""
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior_diag.filtered_means)

    def test_filtered_covariances(self):
        """Check if filtered covariances are close."""
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior_diag.filtered_covariances)

    def test_smoothed_means(self):
        """Check if smoothed means are close."""
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior_diag.smoothed_means)

    def test_smoothed_covariances(self):
        """Check if smoothed covariances are close."""
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior_diag.smoothed_covariances)

    def test_marginal_loglik(self):
        """Check if marginal log likelihoods are close."""
        ser_mll = self.serial_posterior.marginal_loglik
        par_mll = self.parallel_posterior.marginal_loglik
        par_diag_mll = self.parallel_posterior_diag.marginal_loglik
        assert jnp.allclose(ser_mll, par_mll, atol=1e-4 * self.num_timesteps)
        assert jnp.allclose(ser_mll, par_diag_mll, atol=1e-4 * self.num_timesteps)


class TestParallelLGSSMSmootherWithInputs:
    """ Compare parallel and serial lgssm smoothing implementations."""
    
    num_timesteps = 50
    key = jr.PRNGKey(1)

    params, lgssm = make_lgssm_params_with_inputs()   
    params_diag = flatten_diagonal_emission_cov(params)
    inputs = jnp.ones((num_timesteps, 2))
    _, emissions = lgssm_joint_sample(params, key, num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(params, emissions, inputs)
    parallel_posterior_diag = parallel_lgssm_smoother(params_diag, emissions, inputs)

    def test_filtered_means(self):
        """Check if filtered means are close."""
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior_diag.filtered_means)

    def test_filtered_covariances(self):
        """Check if filtered covariances are close."""
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior_diag.filtered_covariances)

    def test_smoothed_means(self):
        """Check if smoothed means are close."""
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior_diag.smoothed_means)

    def test_smoothed_covariances(self):
        """Check if smoothed covariances are close."""
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior_diag.smoothed_covariances)

    def test_marginal_loglik(self):
        """Check if marginal log likelihoods are close."""
        ser_mll = self.serial_posterior.marginal_loglik
        par_mll = self.parallel_posterior.marginal_loglik
        par_diag_mll = self.parallel_posterior_diag.marginal_loglik
        assert jnp.allclose(ser_mll, par_mll, atol=1e-4 * self.num_timesteps)
        assert jnp.allclose(ser_mll, par_diag_mll, atol=1e-4 * self.num_timesteps)


class TestTimeVaryingParallelLGSSMSmoother:
    """Compare parallel and serial time-varying lgssm smoothing implementations.
    
    Vary dynamics weights and observation covariances  with time.
    """
    num_timesteps = 50
    key = jr.PRNGKey(1)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)    
    params_diag = flatten_diagonal_emission_cov(params)
    _, emissions = lgssm_joint_sample(params, key, num_timesteps)

    serial_posterior = serial_lgssm_smoother(params, emissions)
    parallel_posterior = parallel_lgssm_smoother(params, emissions)
    parallel_posterior_diag = parallel_lgssm_smoother(params_diag, emissions)

    def test_filtered_means(self):
        """Check if filtered means are close."""
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior_diag.filtered_means)

    def test_filtered_covariances(self):
        """Check if filtered covariances are close."""
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior_diag.filtered_covariances)

    def test_smoothed_means(self):
        """Check if smoothed means are close."""
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior_diag.smoothed_means)

    def test_smoothed_covariances(self):
        """Check if smoothed covariances are close."""
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior_diag.smoothed_covariances)

    def test_marginal_loglik(self):
        """Check if marginal log likelihoods are close."""
        ser_mll = self.serial_posterior.marginal_loglik
        par_mll = self.parallel_posterior.marginal_loglik
        par_diag_mll = self.parallel_posterior_diag.marginal_loglik
        assert jnp.allclose(ser_mll, par_mll, atol=1e-4 * self.num_timesteps)
        assert jnp.allclose(ser_mll, par_diag_mll, atol=1e-4 * self.num_timesteps)


class TestTimeVaryingParallelLGSSMSampler():
    """Compare parallel and serial lgssm posterior sampling implementations in expectation."""
    
    num_timesteps = 50
    key = jr.PRNGKey(1)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)   
    params_diag = flatten_diagonal_emission_cov(params)
    _, emissions = lgssm_joint_sample(params_diag, key, num_timesteps)

    num_samples = 1000
    serial_keys = jr.split(jr.PRNGKey(2), num_samples)
    parallel_keys = jr.split(jr.PRNGKey(3), num_samples)

    serial_posterior = serial_lgssm_smoother(params, emissions)

    serial_samples = vmap(serial_lgssm_posterior_sample, in_axes=(0,None,None))(
                          serial_keys, params, emissions)
    
    parallel_samples = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None))(
                            parallel_keys, params, emissions)
    
    parallel_samples_diag = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None))(
                                 parallel_keys, params, emissions)

    def test_sampled_means(self):
        """Check if sampled means are close."""
        mc_variance = vmap(jnp.diag)(self.serial_posterior.smoothed_covariances) / self.num_samples
        posterior_mean = self.serial_posterior.smoothed_means
        sample_mean = self.parallel_samples.mean(axis=0)
        sample_mean_diag = self.parallel_samples.mean(axis=0)
        assert jnp.all(abs(sample_mean - posterior_mean) < 6 * jnp.sqrt(mc_variance))
        assert jnp.all(abs(sample_mean_diag - posterior_mean) < 6 * jnp.sqrt(mc_variance))
        
    def test_sampled_covariances(self):
        """Check if sampled covariances are close."""
        # samples have shape (N, T, D): vmap over the T axis, calculate cov over N axis
        serial_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.serial_samples)
        parallel_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        parallel_cov_diag = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        assert allclose(serial_cov, parallel_cov, atol=1e-1)
        assert allclose(serial_cov, parallel_cov_diag, atol=1e-1)