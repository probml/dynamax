from jax import numpy as jnp
from jax import random as jr

from dynamax.linear_gaussian_ssm.inference import lgssm_sample, LGSSMParams
from dynamax.linear_gaussian_ssm.inference import lgssm_smoother as serial_lgssm_smoother
from dynamax.linear_gaussian_ssm.parallel_inference import lgssm_smoother as parallel_lgssm_smoother


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
    input_dim = 1

    lgssm_params = LGSSMParams(
        initial_mean = μ0,
        initial_covariance = Σ0,
        dynamics_matrix = F,
        dynamics_input_weights = jnp.zeros((latent_dim,input_dim)),
        dynamics_bias = jnp.zeros(latent_dim),
        dynamics_covariance = Q,
        emission_matrix = H,
        emission_input_weights = jnp.zeros((observation_dim, input_dim)),
        emission_bias = jnp.zeros(observation_dim),
        emission_covariance = R
    )

    num_timesteps = 50
    inputs = jnp.zeros((num_timesteps,input_dim))

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    _, emissions = lgssm_sample(subkey,lgssm_params,num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(lgssm_params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(lgssm_params, emissions)

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