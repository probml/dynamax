"""
Tests for the linear Gaussian SSM models.
"""

import pytest
import jax.random as jr

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM
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
