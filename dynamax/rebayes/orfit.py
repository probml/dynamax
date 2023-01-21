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
        current_loss_fn = lambda w: loss_fn(w, x, y)
        apply_params_fn = lambda w: apply_fn(w, x)
        g = jacrev(current_loss_fn)(params).squeeze()
        v = jacrev(apply_params_fn)(params).squeeze()
        g_tilde = g - _project_to_columns(U, g)
        v_prime = v - _project_to_columns(U, v)
        
        # Update the U matrix
        u = _normalize(v_prime)
        U = jnp.where(Sigma.min() > u @ v_prime, U, U.at[:, Sigma.argmin()].set(u))
        Sigma = jnp.where(Sigma.min() > u @ v_prime, Sigma, Sigma.at[Sigma.argmin()].set(u.T @ v_prime))

        # Update the parameters
        eta = _stable_division((apply_params_fn(params) - y), (v.T @ g_tilde))
        params = params - eta * g_tilde

        return (params, U, Sigma), (params, U)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases) = scan(_step, carry, jnp.arange(len(inputs)))
    return PosteriorORFitFiltered(filtered_means=filtered_means, filtered_bases=filtered_bases)