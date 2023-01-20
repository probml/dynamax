"""
Implementation of the Orthogonal Recursive Fitting (ORFit) [1] algorithm for online learning.

[1] Min, Y., Ahn, K, & Azizan, N. (2022, July).
One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares.
Retrieved from https://arxiv.org/abs/2207.13853
"""

import jax.numpy as jnp
from jax import jacrev
from jax import vmap
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple

from dynamax.nonlinear_gaussian_ssm.models import FnStateAndInputToEmission
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered


FnStateInputAndOutputToLoss = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"], Float[Array, "output_dim"]], Float[Array, ""]]


# Helper functions
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_projection_matrix = lambda a: _stable_division(a.reshape(-1, 1) @ a.reshape(1, -1), a.T @ a)
_form_projection_matrix = lambda A: jnp.eye(A.shape[0]) - vmap(_projection_matrix, 1)(A).sum(axis=0)
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))
_svd = lambda a: jnp.linalg.svd(a, full_matrices=False)


class ORFitParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    initial_variance: float
    apply_function: FnStateAndInputToEmission
    loss_function: FnStateInputAndOutputToLoss
    memory_size: int


def orthogonal_recursive_fitting(
    model_params: ORFitParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorGSSMFiltered:
    """Vectorized implementation of Orthogonal Recursive Fitting (ORFit) algorithm.

    Args:
        model_params: model parameters.
        emissions: array of observations.
        inputs: array of inputs.

    Returns:
        filtered_posterior: posterior object.
    """
    # Initialize parameters
    initial_mean, initial_variance, apply_fn, loss_fn, memory_limit = model_params
    U, Sigma = jnp.zeros((len(initial_mean), memory_limit)), jnp.zeros((memory_limit,))

    def _step(carry, t):
        params, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]
        current_loss_fn = lambda w: loss_fn(w, x, y)
        apply_params_fn = lambda w: apply_fn(w, x)
        g = jacrev(current_loss_fn)(params)
        v = jacrev(apply_params_fn)(params)
        g_tilde = initial_variance * g - _project_to_columns(U, g.ravel())
        v_prime = initial_variance * v - _project_to_columns(U, v.ravel())
        
        # Update the U matrix
        u = _normalize(v_prime)
        U_tilde, Sigma, _ = _svd(jnp.diag(jnp.append(Sigma, u.ravel() @ v_prime.ravel())))
        U = jnp.hstack((U, u.reshape(-1, 1))) @ U_tilde
        U, Sigma = U[:, :memory_limit], Sigma[:memory_limit]

        # Update the parameters
        eta = _stable_division((apply_params_fn(params) - y), (v.ravel() @ g_tilde.ravel()))
        params = params - eta * g_tilde.ravel()
        cov = _form_projection_matrix(U)

        return (params, U, Sigma), (params, cov)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_covs) = scan(_step, carry, jnp.arange(len(inputs)))
    return PosteriorGSSMFiltered(filtered_means=filtered_means, filtered_covariances=filtered_covs)