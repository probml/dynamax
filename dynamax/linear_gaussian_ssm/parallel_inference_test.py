import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother as serial_lgssm_smoother
from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother

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
        assert jnp.allclose(
                self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means,
                rtol=1e-3
                )

    def test_filtered_covariances(self):
        assert jnp.allclose(
                self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances,
                atol=1e-5,rtol=1e-3
                )

    def test_smoothed_means(self):
        assert jnp.allclose(
                self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means,
                rtol=1e-3
                )

    def test_smoothed_covariances(self):
        assert jnp.allclose(
                self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances,
                atol=1e-5,rtol=1e-3
                )

    def test_marginal_loglik(self):
        assert jnp.allclose(
            self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik, rtol=1e-2
        )