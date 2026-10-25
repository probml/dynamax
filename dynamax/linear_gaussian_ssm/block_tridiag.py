"""
Gaussian distributions over a sequence of states whose precision matrix is block tridiagonal.

Such a Gaussian is stored in information form: a precision block for each time step, a block coupling
each pair of consecutive time steps, and a linear term for each time step. This is the form of the
posterior of a linear Gaussian state space model, and of the approximate posterior over the continuous
states of a switching linear dynamical system. The cost of every routine here grows linearly with the
number of time steps.
"""
from typing import NamedTuple, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.linalg import cho_solve, solve_triangular
from jaxtyping import Array, Float

from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.utils import symmetrize


class BlockTridiagPotentials(NamedTuple):
    """Natural parameters of a Gaussian over a sequence of states with a block-tridiagonal precision matrix.

    :param J_diag: diagonal blocks of the precision matrix, shape (T, D, D)
    :param J_lower: blocks just below the diagonal of the precision matrix, shape (T-1, D, D); each entry couples
        a pair of consecutive time steps
    :param h: linear potentials, shape (T, D)
    """
    J_diag: Float[Array, "num_timesteps state_dim state_dim"]
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]
    h: Float[Array, "num_timesteps state_dim"]


class BlockTridiagPosterior(NamedTuple):
    """Marginals of a Gaussian with block-tridiagonal precision, as returned by :func:`block_tridiag_mvn_smoother`.

    :param J_diag: diagonal blocks of the precision matrix, shape (T, D, D)
    :param J_lower: blocks just below the diagonal of the precision matrix, shape (T-1, D, D)
    :param h: linear potentials, shape (T, D)
    :param log_normalizer: log normalizing constant of the Gaussian
    :param log_det_precision: log determinant of the full precision matrix
    :param filtered_precisions: precisions of the filtering distributions, shape (T, D, D); the sampler reuses them
    :param filtered_linear_potentials: linear potentials of the filtering distributions, shape (T, D)
    :param smoothed_means: mean of each state, shape (T, D)
    :param smoothed_covariances: covariance of each state, shape (T, D, D)
    :param smoothed_cross_covariances: expected outer product of each state with the next one, shape
        (T-1, D, D). The product of the means is not subtracted, as in
        :func:`~dynamax.linear_gaussian_ssm.lgssm_smoother`.
    """
    J_diag: Float[Array, "num_timesteps state_dim state_dim"]
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]
    h: Float[Array, "num_timesteps state_dim"]
    log_normalizer: Scalar
    log_det_precision: Scalar
    filtered_precisions: Float[Array, "num_timesteps state_dim state_dim"]
    filtered_linear_potentials: Float[Array, "num_timesteps state_dim"]
    smoothed_means: Float[Array, "num_timesteps state_dim"]
    smoothed_covariances: Float[Array, "num_timesteps state_dim state_dim"]
    smoothed_cross_covariances: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]


def block_tridiag_mvn_filter(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    h: Float[Array, "num_timesteps state_dim"],
) -> Tuple[Scalar, Scalar,
           Float[Array, "num_timesteps state_dim state_dim"],
           Float[Array, "num_timesteps state_dim"]]:
    """Run the Kalman filter in information form on a Gaussian with block-tridiagonal precision.

    Args:
        J_diag: diagonal blocks of the precision matrix, shape (T, D, D)
        J_lower: blocks just below the diagonal of the precision matrix, shape (T-1, D, D)
        h: linear potentials, shape (T, D)

    Returns:
        log_normalizer: log normalizing constant of the Gaussian
        log_det_precision: log determinant of the full precision matrix
        filtered_precisions: precisions of the filtering distributions, shape (T, D, D)
        filtered_linear_potentials: linear potentials of the filtering distributions, shape (T, D)
    """
    num_timesteps, dim = h.shape
    dtype = jnp.result_type(J_diag, J_lower, h)
    J_diag, J_lower, h = J_diag.astype(dtype), J_lower.astype(dtype), h.astype(dtype)
    J_lower_padded = jnp.concatenate([J_lower, jnp.zeros((1, dim, dim), dtype=dtype)], axis=0)

    def _step(carry, t):
        """Condition on block t, then marginalize x_t out of the pair (x_t, x_{t+1})."""
        J_pred, h_pred, log_normalizer, log_det = carry
        J_cond = J_diag[t] + J_pred
        h_cond = h[t] + h_pred
        chol = jnp.linalg.cholesky(J_cond)
        log_diag = jnp.log(jnp.diag(chol))
        whitened_h = solve_triangular(chol, h_cond, lower=True)                 # chol^{-1} h_cond
        whitened_L = solve_triangular(chol, J_lower_padded[t].T, lower=True)    # chol^{-1} L_t^T
        log_normalizer += 0.5 * dim * jnp.log(2 * jnp.pi) - log_diag.sum() + 0.5 * whitened_h @ whitened_h
        log_det += 2 * log_diag.sum()
        J_pred_next = -whitened_L.T @ whitened_L
        h_pred_next = -whitened_L.T @ whitened_h
        return (J_pred_next, h_pred_next, log_normalizer, log_det), (symmetrize(J_cond), h_cond)

    carry = (jnp.zeros((dim, dim), dtype=h.dtype), jnp.zeros(dim, dtype=h.dtype),
             jnp.zeros((), dtype=h.dtype), jnp.zeros((), dtype=h.dtype))
    (_, _, log_normalizer, log_det), (filtered_precisions, filtered_linear_potentials) = \
        lax.scan(_step, carry, jnp.arange(num_timesteps))
    return log_normalizer, log_det, filtered_precisions, filtered_linear_potentials


def _backward_pass(filtered_precisions, filtered_linear_potentials, J_lower):
    """Backward pass over the filtered potentials, returning the means, covariances and cross-covariances."""
    dim = filtered_linear_potentials.shape[-1]
    eye = jnp.eye(dim, dtype=filtered_linear_potentials.dtype)

    def _last(J_cond, h_cond):
        """Posterior of the last state from its filtered potential."""
        chol = jnp.linalg.cholesky(J_cond)
        mean = cho_solve((chol, True), h_cond)
        cov = symmetrize(cho_solve((chol, True), eye))
        return mean, cov

    def _step(carry, args):
        """Step back one time step.

        Given the next state and all the data, the current state is Gaussian with the filtered precision,
        and a linear potential equal to the filtered one minus the coupling term from the next state.
        """
        mean_next, cov_next = carry
        J_cond, h_cond, L = args
        chol = jnp.linalg.cholesky(J_cond)
        gain = -cho_solve((chol, True), L.T)                      # -J_cond^{-1} L_t^T
        mean = cho_solve((chol, True), h_cond) + gain @ mean_next
        cov = symmetrize(cho_solve((chol, True), eye) + gain @ cov_next @ gain.T)
        cross = gain @ cov_next + jnp.outer(mean, mean_next)   # E[x_t x_{t+1}^T]
        return (mean, cov), (mean, cov, cross)

    mean_last, cov_last = _last(filtered_precisions[-1], filtered_linear_potentials[-1])
    args = (filtered_precisions[:-1], filtered_linear_potentials[:-1], J_lower)
    _, (means, covs, crosses) = lax.scan(_step, (mean_last, cov_last), args, reverse=True)
    return (jnp.concatenate([means, mean_last[None]]),
            jnp.concatenate([covs, cov_last[None]]),
            crosses)


def block_tridiag_mvn_smoother(potentials: BlockTridiagPotentials) -> BlockTridiagPosterior:
    """Compute the mean and covariance of each state of a Gaussian with block-tridiagonal precision.

    Runs :func:`block_tridiag_mvn_filter` forward in time, then a backward pass.

    Args:
        potentials: natural parameters of the Gaussian

    Returns:
        the marginals of each state and of each pair of consecutive states, together with the potentials
        and the filtered potentials
    """
    J_diag, J_lower, h = potentials
    log_normalizer, log_det, filtered_precisions, filtered_linear_potentials = \
        block_tridiag_mvn_filter(J_diag, J_lower, h)
    means, covs, crosses = _backward_pass(filtered_precisions, filtered_linear_potentials, J_lower)
    return BlockTridiagPosterior(J_diag=J_diag, J_lower=J_lower, h=h,
                                 log_normalizer=log_normalizer,
                                 log_det_precision=log_det,
                                 filtered_precisions=filtered_precisions,
                                 filtered_linear_potentials=filtered_linear_potentials,
                                 smoothed_means=means,
                                 smoothed_covariances=covs,
                                 smoothed_cross_covariances=crosses)


def _sample_with_noise(posterior: BlockTridiagPosterior,
                       noise: Float[Array, "num_timesteps state_dim"]) -> Float[Array, "num_timesteps state_dim"]:
    """Draw a sample by backward sampling, using given standard normal noise of shape (T, D) instead of a key."""
    filtered_precisions = posterior.filtered_precisions
    filtered_linear_potentials = posterior.filtered_linear_potentials

    def _draw(J_cond, h_cond, eps):
        """Turn standard normal noise into a draw from the Gaussian with precision J_cond and potential h_cond."""
        chol = jnp.linalg.cholesky(J_cond)
        return cho_solve((chol, True), h_cond) + solve_triangular(chol.T, eps, lower=False)

    x_last = _draw(filtered_precisions[-1], filtered_linear_potentials[-1], noise[-1])

    def _step(x_next, args):
        """Sample the state at one time step given the next state and all the data."""
        J_cond, h_cond, L, eps = args
        x = _draw(J_cond, h_cond - L.T @ x_next, eps)
        return x, x

    _, xs = lax.scan(_step, x_last,
                     (filtered_precisions[:-1], filtered_linear_potentials[:-1], posterior.J_lower, noise[:-1]),
                     reverse=True)
    return jnp.concatenate([xs, x_last[None]])


def block_tridiag_mvn_sample(key: PRNGKeyT,
                             posterior: BlockTridiagPosterior) -> Float[Array, "num_timesteps state_dim"]:
    """Draw one sample of the sequence of states from a Gaussian with block-tridiagonal precision.

    To draw several samples, ``vmap`` over keys.

    Args:
        key: random number generator
        posterior: the Gaussian, as returned by :func:`block_tridiag_mvn_smoother`

    Returns:
        a sample, shape (T, D)
    """
    noise = jr.normal(key, posterior.h.shape, dtype=posterior.h.dtype)
    return _sample_with_noise(posterior, noise)


def block_tridiag_mvn_solve(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    v: Float[Array, "num_timesteps state_dim"],
) -> Float[Array, "num_timesteps state_dim"]:
    """Solve a linear system whose matrix is block tridiagonal and symmetric positive definite.

    The solution is the mean of the Gaussian with this precision matrix and linear potential ``v``.

    Args:
        J_diag: diagonal blocks of the matrix, shape (T, D, D)
        J_lower: blocks just below the diagonal of the matrix, shape (T-1, D, D)
        v: right-hand side, shape (T, D)

    Returns:
        the solution, shape (T, D)
    """
    return block_tridiag_mvn_smoother(BlockTridiagPotentials(J_diag, J_lower, v)).smoothed_means


def block_tridiag_mvn_matvec(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    v: Float[Array, "num_timesteps state_dim"],
) -> Float[Array, "num_timesteps state_dim"]:
    """Multiply a symmetric block-tridiagonal matrix by a vector.

    The blocks above the diagonal are the transposes of the blocks below it.

    Args:
        J_diag: diagonal blocks of the matrix, shape (T, D, D)
        J_lower: blocks just below the diagonal of the matrix, shape (T-1, D, D)
        v: vector, shape (T, D)

    Returns:
        the product, shape (T, D)
    """
    out = jnp.einsum("tij,tj->ti", J_diag, v)
    out = out.at[:-1].add(jnp.einsum("tji,tj->ti", J_lower, v[1:]))   # L_t^T v_{t+1}
    out = out.at[1:].add(jnp.einsum("tij,tj->ti", J_lower, v[:-1]))   # L_{t-1} v_{t-1}
    return out


def block_tridiag_mvn_entropy(log_det_precision: Scalar, num_timesteps: int, state_dim: int) -> Scalar:
    """Compute the entropy of a Gaussian over a sequence of states from the log determinant of its precision.

    Args:
        log_det_precision: log determinant of the full precision matrix, as returned by
            :func:`block_tridiag_mvn_filter` or :func:`block_tridiag_mvn_smoother`
        num_timesteps: number of time steps T
        state_dim: dimension D of each state

    Returns:
        the entropy
    """
    return 0.5 * num_timesteps * state_dim * (1.0 + jnp.log(2 * jnp.pi)) - 0.5 * log_det_precision
