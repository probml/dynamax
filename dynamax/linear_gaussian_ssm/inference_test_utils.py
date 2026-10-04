"""Hand-computed Gaussian references for input and transition timing tests."""

import numpy as np
import jax.numpy as jnp

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params


def make_timing_case(num_timesteps=3):
    """Construct a one- or three-timestep scalar model and its exact reference results."""
    assert num_timesteps in (1, 3)
    inputs = jnp.array([[8.0], [2.0], [4.0]])[:num_timesteps]
    scale = inputs / 2
    params = make_lgssm_params(
        initial_mean=jnp.zeros(1), initial_cov=jnp.eye(1),
        dynamics_weights=scale[:, :, None],
        dynamics_input_weights=scale[:, :, None],
        dynamics_bias=scale,
        # The initial state has no transition, so Q[0] must be unused.
        dynamics_cov=scale[:, :, None].at[0].set(jnp.nan),
        emissions_weights=jnp.eye(1), emissions_cov=jnp.eye(1),
        emissions_input_weights=jnp.array([[0.5]]),
    )
    emissions = jnp.array([[4.0], [4.0], [5.0]])[:num_timesteps]

    # Innovations are [0, 0, -13], with variances [2, 5/2, 27/5].
    # The backward smoothing gains are [1/3, 3/11].
    if num_timesteps == 3:
        means = np.array([-26/27, 1/9, 146/27])
        covariance = np.array([[10/27, 1/9, 2/27], [1/9, 1/3, 2/9], [2/27, 2/9, 22/27]])
        loglik = -0.5 * (3 * np.log(2 * np.pi) + np.log(27) + 169 / (27/5))
    else:
        means, covariance = np.array([0.0]), np.array([[0.5]])
        loglik = -0.5 * np.log(4 * np.pi)
    expected = dict(
        filtered_means=np.array([0, 3, 146/27])[:num_timesteps, None],
        filtered_covariances=np.array([1/2, 3/5, 22/27])[:num_timesteps, None, None],
        predicted_means=np.array([3, 16])[:num_timesteps - 1, None],
        predicted_covariances=np.array([3/2, 22/5])[:num_timesteps - 1, None, None],
        smoothed_means=means[:, None],
        smoothed_covariances=covariance.diagonal()[:, None, None],
        smoothed_cross_covariances=(covariance.diagonal(1) + means[:-1] * means[1:])[:, None, None],
        marginal_loglik=loglik,
        joint_covariance=covariance,
    )
    return params, emissions, inputs, expected


def assert_sample_moments(samples, mean, covariance):
    """Check that samples have approximately the expected mean and covariance."""
    num_samples = len(samples)
    residuals = np.asarray(samples).reshape(num_samples, -1) - np.asarray(mean).ravel()
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), residuals.T).T
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=5 / np.sqrt(num_samples))
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(whitened.shape[1]),
                               atol=6 / np.sqrt(num_samples))
