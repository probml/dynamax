"""
Tests for the linear Gaussian SSM models.
"""
from functools import partial
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
import jax
from jax import jit, vmap
from jax.tree_util import tree_leaves, tree_map
import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
import dynamax.linear_gaussian_ssm.models as models
from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM
from dynamax.utils.utils import monotonically_increasing

NUM_TIMESTEPS = 100

CONFIGS = [
    (LinearGaussianSSM, dict(state_dim=2, emission_dim=10), None),
    (LinearGaussianConjugateSSM, dict(state_dim=2, emission_dim=10), None),
]


@pytest.mark.parametrize("num_forecast_timesteps", [1, 3])
@pytest.mark.parametrize("has_inputs", [False, True])
def test_forecast_matches_expected_moments(num_forecast_timesteps, has_inputs):
    """Check forecast state and observation means and covariances against hand-computed values."""
    model = LinearGaussianSSM(state_dim=1, emission_dim=1, input_dim=int(has_inputs))
    params, _ = model.initialize(
        initial_mean=jnp.zeros(1), initial_covariance=jnp.eye(1),
        dynamics_weights=jnp.array([[0.5]]), dynamics_covariance=jnp.eye(1),
        dynamics_input_weights=jnp.ones((1, int(has_inputs))),
        emission_weights=jnp.eye(1), emission_covariance=jnp.eye(1))
    params = params._replace(dynamics=params.dynamics._replace(
        bias=None, input_weights=params.dynamics.input_weights if has_inputs else None))
    inputs = jnp.zeros((1, 1)) if has_inputs else None
    future_inputs = jnp.array([[10.0], [20.0], [30.0]])[:num_forecast_timesteps] if has_inputs else None

    forecasts = jit(partial(model.forecast, num_forecast_timesteps=num_forecast_timesteps))(
        params, jnp.array([[4.88]]), inputs=inputs, forecast_inputs=future_inputs)

    # Filtering y=4.88 with unit prior/emission variance gives m=2.44, P=0.5.
    # Each future step applies m <- 0.5*m + u and P <- 0.25*P + 1.
    means = [11.22, 25.61, 42.805] if has_inputs else [1.22, 0.61, 0.305]
    expected_means = np.array(means)[:num_forecast_timesteps, None]
    expected_covs = np.array([1.125, 1.28125, 1.3203125])[:num_forecast_timesteps, None, None]
    np.testing.assert_allclose(forecasts[0], expected_means, rtol=1e-6)
    np.testing.assert_allclose(forecasts[1], expected_covs, rtol=1e-6)
    np.testing.assert_allclose(forecasts[2], expected_means, rtol=1e-6)
    np.testing.assert_allclose(forecasts[3], expected_covs + 1, rtol=1e-6)


@pytest.mark.parametrize(["cls", "kwargs", "inputs"], CONFIGS)
def test_sample_and_fit(cls, kwargs, inputs):
    """
    Test that the model can sample and fit the data.
    """
    model = cls(**kwargs)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(0))
    params, param_props = model.initialize(key1)
    states, emissions = model.sample(params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs)
    fitted_params, lps = model.fit_em(params, param_props, emissions, inputs=inputs, num_iters=3)
    assert monotonically_increasing(lps)
    fitted_params, lps = model.fit_sgd(params, param_props, emissions, inputs=inputs, num_epochs=3)


@pytest.mark.parametrize("cls", [LinearGaussianSSM, LinearGaussianConjugateSSM])
def test_em_partially_frozen_params_raises(cls):
    """
    Test that freezing some but not all parameters makes m_step raise.
    """
    model = cls(state_dim=2, emission_dim=3)
    params, props = model.initialize(jr.PRNGKey(0))

    props.dynamics.weights.trainable = False
    # batch_stats=None makes the test fail if m_step uses the statistics before the guard
    with pytest.raises(NotImplementedError):
        model.m_step(params, props, None, None)


@pytest.mark.parametrize("cls", [LinearGaussianSSM, LinearGaussianConjugateSSM])
def test_em_all_frozen_params_unchanged(cls):
    """
    Test that freezing all parameters makes m_step return them unchanged.
    """
    model = cls(state_dim=2, emission_dim=3)
    params, props = model.initialize(jr.PRNGKey(0))
    for group in props:
        for prop in group:
            prop.trainable = False
    new_params, _ = model.m_step(params, props, None, None)
    assert all(tree_leaves(tree_map(jnp.array_equal, new_params, params)))


def test_sgd_supports_frozen_params():
    """
    Test that fit_sgd supports the partial freeze that fit_em refuses, as fit_em's error message advises.
    """
    model = LinearGaussianSSM(state_dim=2, emission_dim=3)
    key1, key2 = jr.split(jr.PRNGKey(0))
    # Generate data with a different dynamics covariance than the starting
    # params (initialize() defaults it to 0.1*I for every key), so the fitting
    # update below is large.
    true_params, _ = model.initialize(key1, dynamics_covariance=0.5 * jnp.eye(2))
    _, emissions = model.sample(true_params, key2, num_timesteps=NUM_TIMESTEPS)

    params, props = model.initialize(jr.PRNGKey(1))
    props.dynamics.weights.trainable = False
    fitted_params, _ = model.fit_sgd(params, props, emissions, num_epochs=5)

    assert jnp.array_equal(fitted_params.dynamics.weights, params.dynamics.weights)
    assert not jnp.allclose(fitted_params.dynamics.cov, params.dynamics.cov)


def test_fit_blocked_gibbs_batched():
    """
    Test that the blocked Gibbs sampler works for multiple observations.
    """
    state_dim = 2
    emission_dim = 3
    num_timesteps = 4
    m_samples = 5
    keys = map(jr.PRNGKey, count())
    m_keys = jr.split(next(keys), num=m_samples)

    model = LinearGaussianConjugateSSM(state_dim, emission_dim)
    params, _ = model.initialize(next(keys))
    _, y_obs = vmap(partial(model.sample, params, num_timesteps=num_timesteps))(m_keys)

    model.fit_blocked_gibbs(next(keys), params, sample_size=6, emissions=y_obs)


def test_e_step_input_statistics(monkeypatch):
    """Check that E-step dynamics sufficient statistics pair z_{t-1} with u_t."""
    model = LinearGaussianSSM(
        state_dim=1, emission_dim=1, input_dim=1)
    inputs = jnp.array([[100.0], [7.0], [-2.0]])
    emissions = jnp.array([[3.0], [-4.0], [8.0]])
    posterior = SimpleNamespace(
        smoothed_means=jnp.array([[2.0], [5.0], [-1.0]]),
        smoothed_covariances=jnp.array([[[0.5]], [[0.25]], [[0.75]]]),
        smoothed_cross_covariances=jnp.array([[[10.1]], [[-5.2]]]),
        marginal_loglik=-12.0)
    monkeypatch.setattr(models, "lgssm_smoother", lambda *args: posterior)

    (_, dynamics, emission), _ = model.e_step(None, emissions, inputs)

    # Dynamics regressors are [2, 7, 1] and [5, -2, 1], with responses 5 and -1.
    # Add the supplied posterior variances and cross moments to these products.
    expected_xx = np.array([[29.75, 4.0, 7.0], [4.0, 53.0, 5.0], [7.0, 5.0, 2.0]])
    expected_xy = np.array([[4.9], [37.0], [4.0]])
    np.testing.assert_allclose(dynamics[0], expected_xx)
    np.testing.assert_allclose(dynamics[1], expected_xy, rtol=1e-6)
    np.testing.assert_allclose(dynamics[2], [[27.0]])
    assert dynamics[3] == 2

    # Emission input statistics still include input[0].
    np.testing.assert_allclose(emission[1][1], [256.0])


def test_gibbs_input_statistics(monkeypatch):
    """Check that Gibbs dynamics sufficient statistics pair z_{t-1} with u_t."""
    model = LinearGaussianConjugateSSM(
        state_dim=1, emission_dim=1, input_dim=1)
    params, _ = model.initialize(jr.PRNGKey(0))
    emissions = jnp.array([[[2.0], [5.0], [-1.0]], [[-3.0], [4.0], [6.0]]])
    inputs = jnp.array([[[100.0], [7.0], [-2.0]], [[200.0], [-5.0], [3.0]]])
    # Fix the sampled latent states to isolate parameter learning from inference.
    monkeypatch.setattr(models, "lgssm_posterior_sample", lambda key, params, emissions, inputs: emissions)
    updates = []

    def record_update(prior, stats):
        """Record the sufficient statistics passed to the parameter update."""
        updates.append(stats)
        # Only the statistics are tested; return the prior so sampling can continue.
        return prior

    monkeypatch.setattr(models, "mniw_posterior_update", record_update)
    with jax.disable_jit():
        model.fit_blocked_gibbs(jr.PRNGKey(1), params, sample_size=1, emissions=emissions, inputs=inputs)

    # Four transitions, with regressors [2,7,1], [5,-2,1], [-3,-5,1], [4,3,1].
    # The inputs 100 and 200 affect no transition and cannot enter these products.
    expected_xx = np.array([[54.0, 31.0, 8.0], [31.0, 87.0, 3.0], [8.0, 3.0, 4.0]])
    expected_xy = np.array([[17.0], [35.0], [14.0]])
    dynamics = updates[0]
    np.testing.assert_allclose(dynamics[0], expected_xx)
    np.testing.assert_allclose(dynamics[1], expected_xy)
    np.testing.assert_allclose(dynamics[2], [[78.0]])
    assert dynamics[3] == 4

    # Emission input statistics still include each sequence's first input.
    np.testing.assert_allclose(updates[1][1][1], [-365.0])
