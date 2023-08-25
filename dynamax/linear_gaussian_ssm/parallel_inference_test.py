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
from dynamax.linear_gaussian_ssm.inference_test import flatten_diagonal_emission_cov


from jax.config import config

config.update("jax_enable_x64", True)


allclose = partial(jnp.allclose, atol=1e-2, rtol=1e-2)


def make_static_lgssm_params():
    latent_dim = 4
    observation_dim = 2
    input_dim = 3

    keys = jr.split(jr.PRNGKey(0), 3)

    dt = 0.1
    F = jnp.eye(4) + dt * jnp.eye(4, k=2)
    B = 0.2 * jr.normal(keys[0], (4, 3))
    b = 0.2 * jnp.arange(4)
    Q = 1.0 * jnp.kron(jnp.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), jnp.eye(2))

    H = jnp.eye(2, 4)
    D = 0.2 * jr.normal(keys[1], (observation_dim, input_dim))
    d = 0.2 * jnp.ones(2)
    R = 0.5**2 * jnp.eye(2)

    μ0 = jnp.array([0.0, 1.0, 1.0, -1.0])
    Σ0 = jnp.eye(4)

    lgssm = LinearGaussianSSM(latent_dim, observation_dim, input_dim)
    params, _ = lgssm.initialize(
        keys[2],
        initial_mean=μ0,
        initial_covariance=Σ0,
        dynamics_weights=F,
        dynamics_input_weights=B,
        dynamics_bias=b,
        dynamics_covariance=Q,
        emission_weights=H,
        emission_input_weights=D,
        emission_bias=d,
        emission_covariance=R,
    )
    return params, lgssm


def make_dynamic_lgssm_params(num_timesteps):
    latent_dim = 4
    observation_dim = 2
    input_dim = 3

    keys = jr.split(jr.PRNGKey(1), 9)

    dt = 0.1

    F = (
        jnp.eye(4)[None]
        + dt * jnp.eye(4, k=2)[None]
        + 0.1 * jr.normal(keys[0], (num_timesteps, latent_dim, latent_dim))
    )
    B = 0.2 * jr.normal(keys[4], (num_timesteps, latent_dim, input_dim))
    b = 0.2 * jr.normal(keys[6], (num_timesteps, latent_dim))
    q_scale = jr.normal(keys[1], (num_timesteps, 1, 1)) ** 2
    Q = q_scale * jnp.kron(jnp.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), jnp.eye(2))[None]

    H = jnp.eye(2, 4)[None] * 0.1 * jr.normal(keys[2], (num_timesteps, observation_dim, latent_dim))
    D = 0.2 * jr.normal(keys[5], (num_timesteps, observation_dim, input_dim))
    d = 0.2 * jr.normal(keys[7], (num_timesteps, observation_dim))
    r_scale = jr.normal(keys[3], (num_timesteps, 1, 1)) ** 2
    R = r_scale * jnp.eye(2)[None]

    μ0 = jnp.array([1.0, -2.0, 1.0, -1.0])
    Σ0 = jnp.eye(latent_dim)

    lgssm = LinearGaussianSSM(latent_dim, observation_dim, input_dim)
    params, _ = lgssm.initialize(
        keys[8],
        initial_mean=μ0,
        initial_covariance=Σ0,
        dynamics_weights=F,
        dynamics_input_weights=B,
        dynamics_bias=b,
        dynamics_covariance=Q,
        emission_weights=H,
        emission_input_weights=D,
        emission_bias=d,
        emission_covariance=R,
    )
    return params, lgssm


class TestParallelLGSSMSmoother:
    """Compare parallel and serial lgssm smoothing implementations."""

    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_static_lgssm_params()
    params_diag = flatten_diagonal_emission_cov(params)
    inputs = jr.normal(keys[0], (num_timesteps, params.dynamics.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(params, emissions, inputs)
    parallel_posterior_diag = parallel_lgssm_smoother(params_diag, emissions, inputs)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior_diag.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior_diag.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior_diag.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior_diag.smoothed_covariances)

    def test_smoothed_cross_covariances(self):
        x = self.serial_posterior.smoothed_cross_covariances
        y = self.parallel_posterior.smoothed_cross_covariances
        z = self.parallel_posterior_diag.smoothed_cross_covariances
        matrix_norm_rel_diff = jnp.linalg.norm(x - y, axis=(1, 2)) / jnp.linalg.norm(x, axis=(1, 2))
        assert allclose(matrix_norm_rel_diff, 0)
        matrix_norm_rel_diff = jnp.linalg.norm(x - z, axis=(1, 2)) / jnp.linalg.norm(x, axis=(1, 2))
        assert allclose(matrix_norm_rel_diff, 0)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik, atol=2e-1)
        assert jnp.allclose(
            self.serial_posterior.marginal_loglik, self.parallel_posterior_diag.marginal_loglik, atol=2e-1
        )


class TestTimeVaryingParallelLGSSMSmoother:
    """Compare parallel and serial time-varying lgssm smoothing implementations.

    Vary dynamics weights and observation covariances  with time.
    """

    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)
    params_diag = flatten_diagonal_emission_cov(params)
    inputs = jr.normal(keys[0], (num_timesteps, params.emissions.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    serial_posterior = serial_lgssm_smoother(params, emissions, inputs)
    parallel_posterior = parallel_lgssm_smoother(params, emissions, inputs)
    parallel_posterior_diag = parallel_lgssm_smoother(params_diag, emissions, inputs)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior_diag.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior_diag.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior_diag.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior_diag.smoothed_covariances)

    def test_smoothed_cross_covariances(self):
        x = self.serial_posterior.smoothed_cross_covariances
        y = self.parallel_posterior.smoothed_cross_covariances
        z = self.parallel_posterior_diag.smoothed_cross_covariances
        matrix_norm_rel_diff = jnp.linalg.norm(x - y, axis=(1, 2)) / jnp.linalg.norm(x, axis=(1, 2))
        assert allclose(matrix_norm_rel_diff, 0)
        matrix_norm_rel_diff = jnp.linalg.norm(x - z, axis=(1, 2)) / jnp.linalg.norm(x, axis=(1, 2))
        assert allclose(matrix_norm_rel_diff, 0)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik, rtol=2e-2)
        assert jnp.allclose(
            self.serial_posterior.marginal_loglik, self.parallel_posterior_diag.marginal_loglik, rtol=2e-2
        )


class TestTimeVaryingParallelLGSSMSampler:
    """Compare parallel and serial lgssm posterior sampling implementations in expectation."""

    num_timesteps = 50
    keys = jr.split(jr.PRNGKey(1), 2)

    params, lgssm = make_dynamic_lgssm_params(num_timesteps)
    params_diag = flatten_diagonal_emission_cov(params)
    inputs = jr.normal(keys[0], (num_timesteps, params.emissions.input_weights.shape[-1]))
    _, emissions = lgssm_joint_sample(params, keys[1], num_timesteps, inputs)

    num_samples = 1000
    serial_keys = jr.split(jr.PRNGKey(2), num_samples)
    parallel_keys = jr.split(jr.PRNGKey(3), num_samples)

    serial_samples = vmap(serial_lgssm_posterior_sample, in_axes=(0, None, None, None))(
        serial_keys, params, emissions, inputs
    )
    parallel_samples = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None, None))(
        parallel_keys, params, emissions, inputs
    )
    parallel_samples_diag = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None, None))(
        parallel_keys, params_diag, emissions, inputs
    )

    def test_sampled_means(self):
        serial_mean = self.serial_samples.mean(axis=0)
        parallel_mean = self.parallel_samples.mean(axis=0)
        parallel_mean_diag = self.parallel_samples.mean(axis=0)
        assert allclose(serial_mean, parallel_mean, atol=1e-1, rtol=1e-1)
        assert allclose(serial_mean, parallel_mean_diag, atol=1e-1, rtol=1e-1)

    def test_sampled_covariances(self):
        # samples have shape (N, T, D): vmap over the T axis, calculate cov over N axis
        serial_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.serial_samples)
        parallel_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        parallel_cov_diag = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        assert allclose(serial_cov, parallel_cov, atol=1e-1)
        assert allclose(serial_cov, parallel_cov_diag, atol=1e-1)
