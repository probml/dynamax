"""
Implementation of the Orthogonal Recursive Fitting (ORFit) [1] algorithm for online learning.

[1] Min, Y., Ahn, K, & Azizan, N. (2022, July).
One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares.
Retrieved from https://arxiv.org/abs/2207.13853
"""

import time

import jax.numpy as jnp
from jax import jacrev
from jax import vmap
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple
import chex

from dynamax.nonlinear_gaussian_ssm.models import FnStateAndInputToEmission


FnStateInputAndOutputToLoss = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"], Float[Array, "output_dim"]], Float[Array, ""]]


# Helper functions
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_projection_matrix = lambda a: _stable_division(a.reshape(-1, 1) @ a.reshape(1, -1), a.T @ a)
_form_projection_matrix = lambda A: jnp.eye(A.shape[0]) - vmap(_projection_matrix, 1)(A).sum(axis=0)
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


class ORFitParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    apply_function: FnStateAndInputToEmission
    loss_function: FnStateInputAndOutputToLoss
    memory_size: int


class PosteriorORFitFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]


def _orfit_condition_on(m, U, Sigma, loss_fn, apply_fn, x, y):
    """Condition on the emission using orfit

    Args:
        m (D_hid): Prior mean.
        U (D_hid, D_mem): Prior basis.
        Sigma (D_mem): Prior singular values.
        loss_fn (Callable): Loss function.
        apply_fn (Callable): Apply function.
        x (D_in): Control input.
        y (D_obs): Emission.

    Returns:
        m_cond (D_hid): Posterior mean.
        U_cond (D_hid, D_mem): Posterior basis.
        Sigma_cond (D_mem): Posterior singular values.
    """    
    l_fn = lambda w: loss_fn(w, x, y)
    f_fn = lambda w: apply_fn(w, x)

    # Compute Jacobians and project out the orthogonal components
    g = jacrev(l_fn)(m).squeeze()
    v = jacrev(f_fn)(m).squeeze()
    g_tilde = g - _project_to_columns(U, g)
    v_prime = v - _project_to_columns(U, v)

    # Update the U matrix
    u = _normalize(v_prime)
    U_cond = jnp.where(Sigma.min() > u @ v_prime, U, U.at[:, Sigma.argmin()].set(u))
    Sigma_cond = jnp.where(Sigma.min() > u @ v_prime, Sigma, Sigma.at[Sigma.argmin()].set(u.T @ v_prime))

    # Update the parameters
    eta = _stable_division((f_fn(m) - y), (v.T @ g_tilde))
    m_cond = m - eta * g_tilde

    return m_cond, U_cond, Sigma_cond


def orthogonal_recursive_fitting(
    model_params: ORFitParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorORFitFiltered:
    """Vectorized implementation of Orthogonal Recursive Fitting (ORFit) algorithm.

    Args:
        model_params: model parameters.
        emissions: array of observations.
        inputs: array of inputs.

    Returns:
        filtered_posterior: posterior object.
    """
    # Initialize parameters
    initial_mean, apply_fn, loss_fn, memory_limit = model_params
    U, Sigma = jnp.zeros((len(initial_mean), memory_limit)), jnp.zeros((memory_limit,))

    def _step(carry, t):
        params, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_params, filtered_U, filtered_Sigma = _orfit_condition_on(params, U, Sigma, loss_fn, apply_fn, x, y)

        return (filtered_params, filtered_U, filtered_Sigma), (filtered_params, filtered_U)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases) = scan(_step, carry, jnp.arange(len(inputs)))
    return PosteriorORFitFiltered(filtered_means=filtered_means, filtered_bases=filtered_bases)


@chex.dataclass
class ORFitBel:
    mean: chex.Array
    basis: chex.Array
    sigma: chex.Array


class RebayesORFit:
    def __init__(
        self,
        orfit_params: ORFitParams,
    ):
        self.update_fn = _orfit_condition_on
        self.mu0 = orfit_params.initial_mean
        self.m = orfit_params.memory_size
        self.U0 = jnp.zeros((len(self.mu0), self.m))
        self.Sigma0 = jnp.zeros((self.m,))
        self.apply_fn = orfit_params.apply_function
        self.loss_fn = orfit_params.loss_function

    def initialize(self):
        return ORFitBel(mean=self.mu0, basis=self.U0, sigma=self.Sigma0)

    def update(self, bel, u, y):
        m, U, Sigma = bel.mean, bel.basis, bel.sigma # prior predictive for hidden state
        m_cond, U_cond, Sigma_cond = self.update_fn(m, U, Sigma, self.loss_fn, self.apply_fn, u, y)
        return ORFitBel(mean=m_cond, basis=U_cond, sigma=Sigma_cond)

    def scan(self, X, Y, callback=None):
        num_timesteps = X.shape[0]
        def step(bel, t):
            bel = self.update(bel, X[t], Y[t])
            if callback is not None:
                out = callback(bel, t, X[t], Y[t])
            return bel, out

        carry = self.initialize()
        bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs