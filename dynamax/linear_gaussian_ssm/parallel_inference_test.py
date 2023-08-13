import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_joint_sample
from dynamax.linear_gaussian_ssm import lgssm_smoother as serial_lgssm_smoother
from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm import lgssm_posterior_sample as serial_lgssm_posterior_sample
from dynamax.linear_gaussian_ssm import parallel_lgssm_posterior_sample



from jax.config import config; config.update("jax_enable_x64", True)



def allclose(x,y, atol=1e-2):
    m = jnp.abs(jnp.max(x-y))
    if m > atol:
        print(m)
        return False
    else:
        return True
    

def make_static_lgssm_params():
    latent_dim = 4
    observation_dim = 2
    input_dim = 3

    dt = 0.1
    F = jnp.eye(4) + dt * jnp.eye(4, k=2)
    b = 0.1 * jnp.arange(4)
    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                          [dt**2/2, dt]]),
                         jnp.eye(2))
                         
    H = jnp.eye(2, 4)
    d = 0.1 * jnp.ones(2)
    R = 0.5 ** 2 * jnp.eye(2)
    μ0 = jnp.array([0.,1.,1.,-1.])
    Σ0 = jnp.eye(4)

    B = jnp.eye(latent_dim, input_dim) * 0
    D = jnp.eye(observation_dim, input_dim) * 0

    lgssm = LinearGaussianSSM(latent_dim, observation_dim, input_dim)
    params, _ = lgssm.initialize(jr.PRNGKey(0),
                             initial_mean=μ0,
                             initial_covariance= Σ0,
                             dynamics_weights=F,
                             dynamics_input_weights=B,
                             dynamics_bias=b,
                             dynamics_covariance=Q,
                             emission_weights=H,
                             emission_input_weights=D,
                             emission_bias=d,
                             emission_covariance=R)
    return params, lgssm
        

def make_dynamic_lgssm_params(num_timesteps):
    latent_dim = 4
    observation_dim = 2
    input_dim = 2

    keys = jr.split(jr.PRNGKey(1), 100)

    key = jr.PRNGKey(0)
    key, key_f, key_r, key_init = jr.split(key, 4)

    dt = 0.1
    f_scale = jr.normal(key_f, (num_timesteps,)) * 0.5
    F = f_scale[:,None,None] * jnp.tile(jnp.eye(latent_dim), (num_timesteps, 1, 1))
    F += dt * jnp.eye(latent_dim, k=2)

    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                                 [dt**2/2, dt]]),
                      jnp.eye(latent_dim // 2))
    assert Q.shape[-1] == latent_dim
    Q = Q[None] * jr.uniform(keys[3], (num_timesteps, 1, 1))

    # H = jnp.eye(observation_dim, latent_dim)
    H = jr.normal(keys[4], (num_timesteps, observation_dim, latent_dim))

    r_scale = jr.normal(key_r, (num_timesteps,)) * 0.1
    R = (r_scale**2)[:,None,None] * jnp.tile(jnp.eye(observation_dim), (num_timesteps, 1, 1))
    
    μ0 = jnp.array([1.,-2.,1.,-1.])
    Σ0 = jnp.eye(latent_dim)

    B = jnp.eye(latent_dim, input_dim)
    D = jnp.eye(observation_dim, input_dim)
    b = jr.normal(keys[0], (num_timesteps, latent_dim))
    d = jr.normal(keys[1], (num_timesteps, observation_dim))

    lgssm = LinearGaussianSSM(latent_dim, observation_dim, input_dim)
    params, _ = lgssm.initialize(key_init,
                                initial_mean=μ0,
                                initial_covariance=Σ0,
                                dynamics_weights=F,
                                dynamics_input_weights=B,
                                dynamics_bias=b,
                                dynamics_covariance=Q,
                                emission_weights=H,
                                emission_input_weights=D,
                                emission_bias=d,
                                emission_covariance=R)
    return params, lgssm


class TestParallelLGSSMSmoother:
    """ Compare parallel and serial lgssm smoothing implementations."""
    
    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_static_lgssm_params()    
    inputs = jr.normal(keys[0], (num_timesteps, params.dynamics.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(params, emissions, inputs)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)

    def test_smoothed_cross_covariances(self):
        assert allclose(
            self.serial_posterior.smoothed_cross_covariances,
            self.parallel_posterior.smoothed_cross_covariances)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik)



class TestTimeVaryingParallelLGSSMSmoother:
    """Compare parallel and serial time-varying lgssm smoothing implementations.
    
    Vary dynamics weights and observation covariances  with time.
    """
    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)    
    inputs = jr.normal(keys[0], (num_timesteps, params.emissions.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(params, emissions, inputs)

    breakpoint()

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)

    def test_smoothed_cross_covariances(self):
        assert allclose(
            self.serial_posterior.smoothed_cross_covariances,
            self.parallel_posterior.smoothed_cross_covariances)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik, atol=1e-1)



class TestTimeVaryingParallelLGSSMSampler():
    """Compare parallel and serial lgssm posterior sampling implementations in expectation."""
    
    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)    
    inputs = jr.normal(keys[0], (num_timesteps, params.emissions.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    num_samples = 1000
    serial_keys = jr.split(jr.PRNGKey(2), num_samples)
    parallel_keys = jr.split(jr.PRNGKey(3), num_samples)

    serial_samples = vmap(serial_lgssm_posterior_sample, in_axes=(0, None, None, None))(
                          serial_keys, params, emissions, inputs)
    
    parallel_samples = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None, None))(
                            parallel_keys, params, emissions, inputs)

    def test_sampled_means(self):
        serial_mean = self.serial_samples.mean(axis=0)
        parallel_mean = self.parallel_samples.mean(axis=0)
        assert allclose(serial_mean, parallel_mean, atol=1e-1)

    def test_sampled_covariances(self):
        # samples have shape (N, T, D): vmap over the T axis, calculate cov over N axis
        serial_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.serial_samples)
        parallel_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        assert allclose(serial_cov, parallel_cov, atol=1e-1)
