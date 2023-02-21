import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_joint_sample
from dynamax.linear_gaussian_ssm import lgssm_smoother as serial_lgssm_smoother
from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother

def allclose(x,y):
    m = jnp.abs(jnp.max(x-y))
    if m > 1e-2:
        print(m)
        return False
    else:
        return True
        
class TestParallelLGSSMSmoother:
    """ Compare parallel and serial lgssm smoothing implementations."""
    num_timesteps=5
    seed=0

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
    model_params, _ = lgssm.initialize(jr.PRNGKey(0),
                             initial_mean=μ0,
                             initial_covariance= Σ0,
                             dynamics_weights=F,
                             dynamics_covariance=Q,
                             emission_weights=H,
                             emission_covariance=R)
    inf_params = model_params

    num_timesteps = 50
    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    _, emissions = lgssm.sample(model_params, subkey, num_timesteps)

    serial_posterior = serial_lgssm_smoother(inf_params, emissions)
    parallel_posterior = parallel_lgssm_smoother(inf_params, emissions)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik)

class TestTimeVaryingParallelLGSSMSmoother:
    """Compare parallel and serial time-varying lgssm smoothing implementations.
    
    Vary dynamics weights and observation covariances  with time.
    """
    seed = 0
    num_timesteps = 50
    latent_dim = 4
    observation_dim = 2

    key = jr.PRNGKey(seed)
    key, key_f, key_q, key_r = jr.split(key, 4)

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

    key, key_init, key_sample = jr.split(key, 3)
    lgssm = LinearGaussianSSM(latent_dim, observation_dim)
    model_params, _ = lgssm.initialize(key_init,
                                       initial_mean=μ0,
                                       initial_covariance=Σ0,
                                       dynamics_weights=F,
                                       dynamics_covariance=Q,
                                       emission_weights=H,
                                       emission_covariance=R)
    inf_params = model_params

    _, emissions = lgssm_joint_sample(model_params, key_sample, num_timesteps)
    
    serial_posterior = serial_lgssm_smoother(inf_params, emissions)
    parallel_posterior = parallel_lgssm_smoother(inf_params, emissions)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik, atol=1e-1)