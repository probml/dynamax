import pytest
from datetime import datetime
from itertools import count
import jax.numpy as jnp
import jax.random as jr
import ssm_jax.linear_gaussian_ssm.models as models
from ssm_jax.utils import add_batch_dim, monotonically_increasing

# def lgssm_test(state_dim=2, emission_dim=10, num_timesteps=100, method='MLE'):
#     keys = map(jr.PRNGKey, count())

#     model = LinearGaussianSSM(state_dim, emission_dim)

#     true_params, props = model.random_initialization(next(keys))
#     true_states, emissions = model.sample(next(keys), num_timesteps)

#     # Fit an LGSSM with EM
#     num_iters = 50
#     test_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
#     marginal_lls = test_model.fit_em(jnp.array([emissions]), num_iters=num_iters, method=method)

#     assert jnp.all(jnp.diff(marginal_lls) > -1e-4)

NUM_TIMESTEPS = 100

CONFIGS = [
    (models.LinearGaussianSSM, dict(state_dim=2, emission_dim=10), dict()),
]

@pytest.mark.parametrize(["cls", "kwargs", "covariates"], CONFIGS)
def test_sample(cls, kwargs, covariates):
    model = cls(**kwargs)
    key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    params, param_props = model.random_initialization(key1)
    states, emissions = model.sample(params, key2, num_timesteps=NUM_TIMESTEPS, **covariates)


@pytest.mark.parametrize(["cls", "kwargs", "covariates"], CONFIGS)
def test_fit_em(cls, kwargs, covariates):
    model = cls(**kwargs)
    key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    params, param_props = model.random_initialization(key1)
    states, emissions = model.sample(params, key2, num_timesteps=NUM_TIMESTEPS, **covariates)
    fitted_params, lps = model.fit_em(params, param_props, add_batch_dim(emissions), **add_batch_dim(covariates), num_iters=10)
    assert monotonically_increasing(lps)


@pytest.mark.parametrize(["cls", "kwargs", "covariates"], CONFIGS)
def test_fit_sgd(cls, kwargs, covariates):
    model = cls(**kwargs)
    key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    params, param_props = model.random_initialization(key1)
    states, emissions = model.sample(params, key2, num_timesteps=NUM_TIMESTEPS, **covariates)
    fitted_params, lps = model.fit_sgd(params, param_props, add_batch_dim(emissions), **add_batch_dim(covariates), num_epochs=10)


# if __name__ == "__main__":

#     print("Test the MLE estimation with EM algorithm ... ")
#     lgssm_test(method='MLE')
#     print("Test of EM algorithm completed.")

#     print("Test the MAP estimation with EMAP algorithm ... ")
#     lgssm_test(method='MAP')
#     print("Test of EMAP algorithm completed.")
