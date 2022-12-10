from jax import numpy as jnp
from jax import lax, jacrev, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN
import chex
from typing import Callable

from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.dekf.diagonal_inference import _fully_decoupled_ekf_condition_on
from dynamax.generalized_gaussian_ssm.dekf.diagonal_inference import _variational_diagonal_ekf_condition_on
from dynamax.generalized_gaussian_ssm.dekf.diagonal_inference import _full_covariance_condition_on


@chex.dataclass
class GaussianBel:
    mean: chex.Array
    cov: chex.Array


class RebayesEKF:
    def __init__(
        self,
        ssm_params: ParamsGGSSM,
        method: str
    ):
        if method == 'fcekf':
            self.update_fn = _full_covariance_condition_on
            Sigma0 = ssm_params.initial_covariance
            Q = ssm_params.dynamics_covariance
        elif method == 'vdekf':
            self.update_fn = _variational_diagonal_ekf_condition_on
            Sigma0 = jnp.diag(ssm_params.ssm_params.initial_covariance)
            Q = jnp.diag(ssm_params.dynamics_covariance)
        elif method == 'fdekf':
            self.update_fn = _fully_decoupled_ekf_condition_on
            Sigma0 = jnp.diag(ssm_params.ssm_params.initial_covariance)
            Q = jnp.diag(ssm_params.dynamics_covariance)
        else:
            raise ValueError('unknown method ', method)
        self.mu0 = ssm_params.initial_mean
        self.Sigma0 = Sigma0
        self.Q = Q
        self.mean_Y = ssm_params.emission_mean_function
        self.cov_Y = ssm_params.emission_cov_function

    def initialize(self):
        return GaussianBel(mean=self.mu0, cov=self.Sigma0)

    def update(self, bel, u, y):
        m, P = bel.mean, bel.cov + self.Q # prior predictive for hidden state
        mu, Sigma = self.update_fn(m, P, self.mean_Y, self.cov_Y, u, y, num_iter=1)
        return GaussianBel(mean=mu, cov=Sigma)

    def scan(self, X, Y, callback=None):
        num_timesteps = X.shape[0]
        def step(bel, t):
            bel = self.update(bel, X[t], Y[t])
            if callback is not None:
                out = callback(bel, t, X[t], Y[t])
            return bel, out

        carry = self.initialize()
        bel, outputs = lax.scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs
 