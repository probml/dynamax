import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_flatten
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import dynamax.structural_time_series.models.structural_time_series as sts


def sample_time_series(initial_mean, F, H, Q, R, num_timesteps, key):

    dim_state = Q.shape[-1]
    dynamics_noise = MVN(loc=jnp.zeros(dim_state), covariance_matrix=Q)
    emission_noise = MVN(loc=jnp.zeros(1), covariance_matrix=R)

    def step(cur_state, key):
        key1, key2 = jr.split(key)
        cur_emission = H @ cur_state + emission_noise.sample(seed=key1)
        next_state = F @ cur_state + dynamics_noise.sample(seed=key2)
        return next_state, (H @ cur_state, cur_emission)

    initial_state = initial_mean + dynamics_noise.sample(seed=key)
    keys = jr.split(key, num_timesteps)
    _, (states, time_series) = lax.scan(step, initial_state, keys)

    return states, time_series


def test_local_linear(seed=1, num_timesteps=100):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 4)

    # Synthetic data using STS model with only local_linear_trend component
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])

    var = jnp.exp(jnp.array([-2, -1, 0.]) + jr.normal(keys[0], shape=(3,)))
    Q = jnp.array([[var[0], 0],
                   [0, var[1]]])
    R = jnp.array([[var[2]]])
    params_true = var

    initial_mean = jr.normal(keys[1], shape=(2,))
    states, observed_time_series = sample_time_series(initial_mean, F, H, Q, R, num_timesteps, keys[2])

    # Model fit using MLE
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    model = sts.StructuralTimeSeries(
        [trend], observation_distribution_family='Gaussian', observed_time_series=observed_time_series)

    optimal_params, losses = model.fit_mle(observed_time_series, key=keys[3])
    params_fitted = optimal_params['dynamics_covariances'].values(), optimal_params['']

    assert jnp.allclose(params_true, params_fitted)


def test_dummy_seasonal(key=10, num_seasons=12, num_rounds=10):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 4)

    # Synthetic data using STS model with only local_linear_trend component
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])

    var = jnp.exp(jnp.array([-2, -1, 0.]) + jr.normal(keys[0], shape=(3,)))
    Q = jnp.array([[var[0], 0],
                   [0, var[1]]])
    R = jnp.array([[var[2]]])
    params_true = var

    initial_mean = jr.normal(keys[1], shape=(2,))
    states, observed_time_series = sample_time_series(initial_mean, F, H, Q, R, num_timesteps, keys[2])

    # Model fit using MLE
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    model = sts.StructuralTimeSeries(
        [trend], observation_distribution_family='Gaussian', observed_time_series=observed_time_series)

    optimal_params, losses = model.fit_mle(observed_time_series, key=keys[3])
    params_fitted = optimal_params['dynamics_covariances'].values(), optimal_params['']

    assert jnp.allclose(params_true, params_fitted)


def test_trigonometric_seasonal(key=10, num_seasons=12, num_rounds=10):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 4)

    # Synthetic data using STS model with only local_linear_trend component
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])

    var = jnp.exp(jnp.array([-2, -1, 0.]) + jr.normal(keys[0], shape=(3,)))
    Q = jnp.array([[var[0], 0],
                   [0, var[1]]])
    R = jnp.array([[var[2]]])
    params_true = var

    initial_mean = jr.normal(keys[1], shape=(2,))
    states, observed_time_series = sample_time_series(initial_mean, F, H, Q, R, num_timesteps, keys[2])

    # Model fit using MLE
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    model = sts.StructuralTimeSeries(
        [trend], observation_distribution_family='Gaussian', observed_time_series=observed_time_series)

    optimal_params, losses = model.fit_mle(observed_time_series, key=keys[3])
    params_fitted = optimal_params['dynamics_covariances'].values(), optimal_params['']

    assert jnp.allclose(params_true, params_fitted)


def test_linear_regression(key=10, num_timesteps=100):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 4)

    # Synthetic data using STS model with only local_linear_trend component
    F = jnp.array([[1, 1],
                   [0, 1]])
    H = jnp.array([[1, 0]])

    var = jnp.exp(jnp.array([-2, -1, 0.]) + jr.normal(keys[0], shape=(3,)))
    Q = jnp.array([[var[0], 0],
                   [0, var[1]]])
    R = jnp.array([[var[2]]])
    params_true = var

    initial_mean = jr.normal(keys[1], shape=(2,))
    states, observed_time_series = sample_time_series(initial_mean, F, H, Q, R, num_timesteps, keys[2])

    # Model fit using MLE
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    model = sts.StructuralTimeSeries(
        [trend], observation_distribution_family='Gaussian', observed_time_series=observed_time_series)

    optimal_params, losses = model.fit_mle(observed_time_series, key=keys[3])
    params_fitted = optimal_params['dynamics_covariances'].values(), optimal_params['']

    assert jnp.allclose(params_true, params_fitted)
