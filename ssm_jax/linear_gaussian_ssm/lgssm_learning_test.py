import jax.numpy as jnp
import jax.random as jr
from jax import jit
from itertools import count
import matplotlib.pyplot as plt

from ssm_jax.linear_gaussian_ssm.models import LinearGaussianSSM


def lgssm_test(state_dim=2, emission_dim=10, num_timesteps=100, method='MLE'):
    keys = map(jr.PRNGKey, count())

    true_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    true_states, emissions = true_model.sample(next(keys), num_timesteps)

    # Fit an LGSSM with EM
    num_iters = 50
    test_model = LinearGaussianSSM.random_initialization(next(keys), state_dim, emission_dim)
    marginal_lls = test_model.fit_em(jnp.array([emissions]), num_iters=num_iters, method=method)

    assert jnp.all(jnp.diff(marginal_lls) > -1e-4)

if __name__ == "__main__":

    print("Test the MLE estimation with EM algorithm ... ")
    lgssm_test(method='MLE')
    print("Test of EM algorithm completed.")

    print("Test the MAP estimation with EMAP algorithm ... ")
    lgssm_test(method='MAP')
    print("Test of EMAP algorithm completed.")

