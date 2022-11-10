import pytest
from datetime import datetime
import jax.random as jr
import jax.numpy as jnp
from dynamax.linear_gaussian_ssm.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm.linear_gaussian_ssm_conjugate import LinearGaussianConjugateSSM
from dynamax.utils import monotonically_increasing

from dynamax.utils import has_tpu

NUM_TIMESTEPS = 100

CONFIGS = [
    (LinearGaussianSSM, dict(state_dim=2, emission_dim=10), None),
    (LinearGaussianConjugateSSM, dict(state_dim=2, emission_dim=10), None),
]

@pytest.mark.parametrize(["cls", "kwargs", "inputs"], CONFIGS)
def s_test_sample_and_fit(cls, kwargs, inputs):
    model = cls(**kwargs)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(0))
    params, param_props = model.initialize(key1)
    states, emissions = model.sample(params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs)
    fitted_params, lps = model.fit_em(params, param_props, emissions, inputs=inputs, num_iters=3)
    if not has_tpu():
        assert monotonically_increasing(lps) # fails on TPU
    fitted_params, lps = model.fit_sgd(params, param_props, emissions, inputs=inputs, num_epochs=3)

def test_type_checking():
    model = LinearGaussianSSM(state_dim=2, emission_dim=10)
    key1, key2 = jr.split(jr.PRNGKey(0))
    mu = jnp.zeros(5) # wrong shape
    params, param_props = model.initialize(key1, initial_mean=mu)
    print(params.initial.mean.shape)
    print(params)
    assert params.initial.mean.shape == model.state_dim

