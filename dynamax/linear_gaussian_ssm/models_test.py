"""
Tests for the linear Gaussian SSM models.
"""
from functools import partial
from itertools import count, product

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax import tree
import pytest

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM
from dynamax.utils.utils import monotonically_increasing

NUM_TIMESTEPS = 100

CONFIGS = [
    (LinearGaussianSSM, dict(state_dim=2, emission_dim=10), None),
    (LinearGaussianConjugateSSM, dict(state_dim=2, emission_dim=10), None),
]

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

@pytest.mark.parametrize(["has_dynamics_bias", "has_emissions_bias"], product([True, False], repeat=2))
def test_inhomogeneous_lgcssm(has_dynamics_bias, has_emissions_bias):
    """
    Test a LinearGaussianConjugateSSM with time-varying dynamics and emission model.
    """
    state_dim = 2
    emission_dim = 3
    num_timesteps = 4
    keys = map(jr.PRNGKey, count())
    kwargs = {
        "state_dim": state_dim,
        "emission_dim": emission_dim,
        "has_dynamics_bias": has_dynamics_bias,
        "has_emissions_bias": has_emissions_bias,
    }
    model = LinearGaussianConjugateSSM(**kwargs)
    params, param_props = model.initialize(jr.PRNGKey(0))
    # Repeat the parameters for each timestep.
    inhomogeneous_dynamics = tree.map(
        lambda x: jnp.repeat(x[None], num_timesteps - 1, axis=0), params.dynamics,
    )
    inhomogeneous_emissions = tree.map(
        lambda x: jnp.repeat(x[None], num_timesteps, axis=0), params.emissions,
    )

    _, emissions = model.sample(params, next(keys), num_timesteps=num_timesteps)
    inhomogeneous_params = ParamsLGSSM(
        initial=params.initial,
        dynamics=inhomogeneous_dynamics,
        emissions=inhomogeneous_emissions,
    )
    params_trace = model.fit_blocked_gibbs(
        next(keys),
        inhomogeneous_params,
        sample_size=5,
        emissions=emissions,
    )

    # Arbitrarily check the last set of parameters from the Markov chain.
    last_params = tree.map(lambda x: x[-1], params_trace)
    assert last_params.initial.mean.shape == (state_dim,)
    assert last_params.initial.cov.shape == (state_dim, state_dim)
    assert last_params.dynamics.weights.shape == (num_timesteps - 1, state_dim, state_dim)
    assert last_params.emissions.weights.shape == (num_timesteps, emission_dim, state_dim)
    assert last_params.dynamics.bias.shape == (num_timesteps - 1, state_dim)
    assert last_params.emissions.bias.shape == (num_timesteps, emission_dim)
    assert last_params.dynamics.cov.shape == (num_timesteps - 1, state_dim, state_dim)
    assert last_params.emissions.cov.shape == (num_timesteps, emission_dim, emission_dim)
