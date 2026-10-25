r"""
Gaussian primitives for a sequence :math:`x_{1:T}` whose precision matrix is block tridiagonal.

The distribution is stored in information form,

.. math::
    \log q(x) = -\tfrac{1}{2} \sum_t x_t^\top J_t x_t - \sum_{t < T} x_{t+1}^\top L_t x_t
                + \sum_t h_t^\top x_t - \log Z,

with diagonal blocks :math:`J_t` (``J_diag``), lower off-diagonal blocks :math:`L_t`
(``J_lower[t]`` is block :math:`(t+1, t)` of the precision matrix) and linear potentials
:math:`h_t`. This is the posterior of every linear Gaussian chain, and the Laplace
approximation of the continuous states of a switching linear dynamical system. All routines
cost :math:`O(T D^3)` and run as a single ``lax.scan``.
"""
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.scipy.linalg import cho_solve, solve_triangular
from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, _get_params, preprocess_params_and_inputs
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.utils import symmetrize


class BlockTridiagPotentials(NamedTuple):
    r"""Natural parameters of a Gaussian over :math:`x_{1:T}` with block-tridiagonal precision.

    :param J_diag: diagonal precision blocks :math:`J_t`
    :param J_lower: lower off-diagonal precision blocks :math:`L_t`, block :math:`(t+1, t)`
    :param h: linear potentials :math:`h_t`
    """
    J_diag: Float[Array, "num_timesteps state_dim state_dim"]
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]
    h: Float[Array, "num_timesteps state_dim"]


class BlockTridiagPosterior(NamedTuple):
    r"""Marginals of a block-tridiagonal Gaussian, as returned by :func:`block_tridiag_mvn_smoother`.

    :param log_normalizer: :math:`\log Z`
    :param log_det_precision: :math:`\log |J|` of the full :math:`TD \times TD` precision matrix
    :param filtered_precisions: precisions of the filtering distributions, for the sampler
    :param filtered_linear_potentials: linear potentials of the filtering distributions
    :param smoothed_means: :math:`\mathbb{E}[x_t]`
    :param smoothed_covariances: :math:`\mathrm{Cov}[x_t]`
    :param smoothed_cross_covariances: :math:`\mathbb{E}[x_t x_{t+1}^\top]`, uncentered, as in
        :func:`~dynamax.linear_gaussian_ssm.lgssm_smoother`
    """
    log_normalizer: Scalar
    log_det_precision: Scalar
    filtered_precisions: Float[Array, "num_timesteps state_dim state_dim"]
    filtered_linear_potentials: Float[Array, "num_timesteps state_dim"]
    smoothed_means: Float[Array, "num_timesteps state_dim"]
    smoothed_covariances: Float[Array, "num_timesteps state_dim state_dim"]
    smoothed_cross_covariances: Float[Array, "num_timesteps_minus_1 state_dim state_dim"]


def _cholesky(precision: Array, jitter: float) -> Array:
    """Lower Cholesky factor of a symmetrized precision block, optionally jittered."""
    dim = precision.shape[-1]
    return jnp.linalg.cholesky(symmetrize(precision) + jitter * jnp.eye(dim, dtype=precision.dtype))


def block_tridiag_mvn_filter(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    h: Float[Array, "num_timesteps state_dim"],
    jitter: float = 0.0,
) -> Tuple[Scalar, Scalar,
           Float[Array, "num_timesteps state_dim state_dim"],
           Float[Array, "num_timesteps state_dim"]]:
    r"""Information-form Kalman filter for a block-tridiagonal Gaussian.

    One ``lax.scan`` over :math:`t = 0, \ldots, T-1`. Step :math:`t` conditions the predicted
    potential on block :math:`t`, :math:`J^c_t = J_t + J^p_t`, :math:`h^c_t = h_t + h^p_t`, and
    marginalizes :math:`x_t` out of the pair :math:`(x_t, x_{t+1})`,
    :math:`J^p_{t+1} = -L_t (J^c_t)^{-1} L_t^\top`, :math:`h^p_{t+1} = -L_t (J^c_t)^{-1} h^c_t`,
    with ``J_lower`` padded by one zero block so the last step predicts nothing.

    Args:
        J_diag: diagonal precision blocks.
        J_lower: lower off-diagonal precision blocks (block :math:`(t+1, t)`).
        h: linear potentials.
        jitter: added to the diagonal of every conditioned precision block before its Cholesky
            factorization. A positive value changes the distribution (it is a bias, not a
            correction); the default 0 is exact.

    Returns:
        log_normalizer: :math:`\log Z`.
        log_det_precision: :math:`\log |J| = 2 \sum_t \sum_i \log \mathrm{chol}(J^c_t)_{ii}`.
        filtered_precisions: :math:`J^c_t`, shape (T, D, D).
        filtered_linear_potentials: :math:`h^c_t`, shape (T, D).
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
        chol = _cholesky(J_cond, jitter)
        log_diag = jnp.log(jnp.diag(chol))
        whitened_h = solve_triangular(chol, h_cond, lower=True)                 # chol^{-1} h^c
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


def _backward_pass(filtered_precisions, filtered_linear_potentials, J_lower, jitter, with_covariances):
    """Backward recursion over the filtered potentials.

    Returns means only (for the solver) or means, covariances and cross-covariances.
    """
    dim = filtered_linear_potentials.shape[-1]
    eye = jnp.eye(dim, dtype=filtered_linear_potentials.dtype)

    def _last(J_cond, h_cond):
        """Posterior of the last state from its filtered potential."""
        chol = _cholesky(J_cond, jitter)
        mean = cho_solve((chol, True), h_cond)
        cov = symmetrize(cho_solve((chol, True), eye))
        return mean, cov

    def _step(carry, args):
        """p(x_t | x_{t+1}, all data) = N(J_c^{-1} (h_c - L_t^T x_{t+1}), J_c^{-1})."""
        mean_next, cov_next = carry
        J_cond, h_cond, L = args
        chol = _cholesky(J_cond, jitter)
        gain = -cho_solve((chol, True), L.T)                      # G_t = -J_c^{-1} L_t^T
        mean = cho_solve((chol, True), h_cond) + gain @ mean_next
        if with_covariances:
            cov = symmetrize(cho_solve((chol, True), eye) + gain @ cov_next @ gain.T)
            cross = gain @ cov_next + jnp.outer(mean, mean_next)   # E[x_t x_{t+1}^T]
            return (mean, cov), (mean, cov, cross)
        return (mean, cov_next), mean

    mean_last, cov_last = _last(filtered_precisions[-1], filtered_linear_potentials[-1])
    args = (filtered_precisions[:-1], filtered_linear_potentials[:-1], J_lower)
    _, outputs = lax.scan(_step, (mean_last, cov_last), args, reverse=True)
    if with_covariances:
        means, covs, crosses = outputs
        return (jnp.concatenate([means, mean_last[None]]),
                jnp.concatenate([covs, cov_last[None]]),
                crosses)
    return jnp.concatenate([outputs, mean_last[None]])


def block_tridiag_mvn_smoother(potentials: BlockTridiagPotentials, jitter: float = 0.0) -> BlockTridiagPosterior:
    r"""Means, covariances and cross-covariances of a block-tridiagonal Gaussian.

    A forward :func:`block_tridiag_mvn_filter` followed by a backward ``lax.scan`` over
    :math:`t = T-2, \ldots, 0` through the conditionals
    :math:`p(x_t \mid x_{t+1}, \cdot) = \mathcal{N}((J^c_t)^{-1}(h^c_t - L_t^\top x_{t+1}), (J^c_t)^{-1})`:
    with :math:`G_t = -(J^c_t)^{-1} L_t^\top`,
    :math:`\mu_t = (J^c_t)^{-1} h^c_t + G_t \mu_{t+1}`,
    :math:`\Sigma_t = (J^c_t)^{-1} + G_t \Sigma_{t+1} G_t^\top` and
    :math:`\mathbb{E}[x_t x_{t+1}^\top] = G_t \Sigma_{t+1} + \mu_t \mu_{t+1}^\top`.

    Args:
        potentials: the block-tridiagonal natural parameters.
        jitter: see :func:`block_tridiag_mvn_filter`.

    Returns:
        the posterior marginals and the filtered potentials.
    """
    J_diag, J_lower, h = potentials
    log_normalizer, log_det, filtered_precisions, filtered_linear_potentials = \
        block_tridiag_mvn_filter(J_diag, J_lower, h, jitter)
    means, covs, crosses = _backward_pass(filtered_precisions, filtered_linear_potentials, J_lower, jitter, True)
    return BlockTridiagPosterior(log_normalizer=log_normalizer,
                                 log_det_precision=log_det,
                                 filtered_precisions=filtered_precisions,
                                 filtered_linear_potentials=filtered_linear_potentials,
                                 smoothed_means=means,
                                 smoothed_covariances=covs,
                                 smoothed_cross_covariances=crosses)


def _sample_with_noise(potentials: BlockTridiagPotentials,
                       noise: Float[Array, "num_timesteps state_dim"],
                       filtered: Optional[Tuple[Array, Array]] = None,
                       jitter: float = 0.0) -> Float[Array, "num_timesteps state_dim"]:
    """Deterministic backward sampling given standard normal noise of shape (T, D)."""
    J_diag, J_lower, h = potentials
    if filtered is None:
        _, _, filtered_precisions, filtered_linear_potentials = block_tridiag_mvn_filter(J_diag, J_lower, h, jitter)
    else:
        filtered_precisions, filtered_linear_potentials = filtered

    def _draw(J_cond, h_cond, eps):
        """x = J_c^{-1} h_c + chol(J_c)^{-T} eps has mean J_c^{-1} h_c and covariance J_c^{-1}."""
        chol = _cholesky(J_cond, jitter)
        return cho_solve((chol, True), h_cond) + solve_triangular(chol.T, eps, lower=False)

    x_last = _draw(filtered_precisions[-1], filtered_linear_potentials[-1], noise[-1])

    def _step(x_next, args):
        """Sample x_t from p(x_t | x_{t+1}, all data)."""
        J_cond, h_cond, L, eps = args
        x = _draw(J_cond, h_cond - L.T @ x_next, eps)
        return x, x

    _, xs = lax.scan(_step, x_last,
                     (filtered_precisions[:-1], filtered_linear_potentials[:-1], J_lower, noise[:-1]),
                     reverse=True)
    return jnp.concatenate([xs, x_last[None]])


def block_tridiag_mvn_sample(key: PRNGKeyT,
                             potentials: BlockTridiagPotentials,
                             filtered: Optional[Tuple[Array, Array]] = None,
                             jitter: float = 0.0) -> Float[Array, "num_timesteps state_dim"]:
    r"""Draw one sequence :math:`x_{1:T}` from a block-tridiagonal Gaussian.

    Forward filter (skipped when ``filtered`` is given), then backward sampling through the
    conditionals :math:`x_t = (J^c_t)^{-1}(h^c_t - L_t^\top x_{t+1}) + \mathrm{chol}(J^c_t)^{-\top} \epsilon_t`.
    ``vmap`` over keys for several samples.

    Args:
        key: PRNG key.
        potentials: the block-tridiagonal natural parameters.
        filtered: optional ``(filtered_precisions, filtered_linear_potentials)`` from a previous
            filter or smoother call on the same potentials.
        jitter: see :func:`block_tridiag_mvn_filter`.

    Returns:
        a sample of shape (T, D).
    """
    noise = jr.normal(key, potentials.h.shape, dtype=potentials.h.dtype)
    return _sample_with_noise(potentials, noise, filtered, jitter)


def block_tridiag_mvn_solve(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    v: Float[Array, "num_timesteps state_dim"],
    jitter: float = 0.0,
) -> Float[Array, "num_timesteps state_dim"]:
    r"""Solve :math:`J x = v` for a block-tridiagonal :math:`J` in :math:`O(T D^3)`.

    The solution is the mean of the Gaussian with precision :math:`J` and linear potential
    :math:`v`, so this is a filter with ``h = v`` followed by the backward mean recursion.
    """
    _, _, filtered_precisions, filtered_linear_potentials = block_tridiag_mvn_filter(J_diag, J_lower, v, jitter)
    return _backward_pass(filtered_precisions, filtered_linear_potentials, J_lower, jitter, False)


def block_tridiag_mvn_matvec(
    J_diag: Float[Array, "num_timesteps state_dim state_dim"],
    J_lower: Float[Array, "num_timesteps_minus_1 state_dim state_dim"],
    v: Float[Array, "num_timesteps state_dim"],
) -> Float[Array, "num_timesteps state_dim"]:
    r"""The product :math:`J v` for a block-tridiagonal :math:`J`.

    :math:`(Jv)_t = J_t v_t + L_t^\top v_{t+1} + L_{t-1} v_{t-1}`, with the off-diagonal terms
    absent at the ends.
    """
    out = jnp.einsum("tij,tj->ti", J_diag, v)
    out = out.at[:-1].add(jnp.einsum("tji,tj->ti", J_lower, v[1:]))   # L_t^T v_{t+1}
    out = out.at[1:].add(jnp.einsum("tij,tj->ti", J_lower, v[:-1]))   # L_{t-1} v_{t-1}
    return out


def block_tridiag_mvn_entropy(log_det_precision: Scalar, num_timesteps: int, state_dim: int) -> Scalar:
    r"""Entropy of a :math:`TD`-dimensional Gaussian from the log determinant of its precision,
    :math:`\mathcal{H} = \tfrac{TD}{2}(1 + \log 2\pi) - \tfrac{1}{2} \log |J|`."""
    return 0.5 * num_timesteps * state_dim * (1.0 + jnp.log(2 * jnp.pi)) - 0.5 * log_det_precision


def _inverse_and_logdet(cov: Array) -> Tuple[Array, Array]:
    """Inverse and log determinant of a symmetric positive definite matrix via Cholesky."""
    chol = _cholesky(cov, 0.0)
    eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    return symmetrize(cho_solve((chol, True), eye)), 2 * jnp.log(jnp.diag(chol)).sum()


def lgssm_to_block_tridiag(params: ParamsLGSSM,
                           emissions: Float[Array, "num_timesteps emission_dim"],
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
                           ) -> Tuple[BlockTridiagPotentials, Scalar]:
    r"""Natural parameters of :math:`p(x_{1:T} \mid y_{1:T}, u_{1:T})` for a linear Gaussian SSM.

    Uses dynamax's convention :math:`x_{t+1} = F_{t+1} x_t + B_{t+1} u_{t+1} + b_{t+1} + \mathcal{N}(0, Q_{t+1})`
    and :math:`y_t = H_t x_t + D_t u_t + d_t + \mathcal{N}(0, R_t)`: time-varying dynamics
    parameters have one entry per timestep and entry 0 is never read. With
    :math:`c_t = B_t u_t + b_t` and :math:`r_t = y_t - D_t u_t - d_t`,

    .. math::
        J_t &= H_t^\top R_t^{-1} H_t + [t=0]\, \Sigma_0^{-1} + [t<T-1]\, F_{t+1}^\top Q_{t+1}^{-1} F_{t+1} + [t>0]\, Q_t^{-1}, \\
        L_t &= -Q_{t+1}^{-1} F_{t+1}, \\
        h_t &= H_t^\top R_t^{-1} r_t + [t=0]\, \Sigma_0^{-1} m_0 - [t<T-1]\, F_{t+1}^\top Q_{t+1}^{-1} c_{t+1} + [t>0]\, Q_t^{-1} c_t.

    Args:
        params: model parameters (``None`` biases and input weights are treated as zero).
        emissions: observations, shape (T, N).
        inputs: optional inputs, shape (T, M).

    Returns:
        potentials: the block-tridiagonal natural parameters.
        log_constant: the terms of :math:`\log p(x, y)` that do not depend on :math:`x`, so that
            ``log_normalizer + log_constant`` equals the marginal log likelihood from
            :func:`~dynamax.linear_gaussian_ssm.lgssm_filter`.
    """
    num_timesteps = emissions.shape[0]
    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)
    state_dim = params.initial.mean.shape[0]
    emission_dim = emissions.shape[1]

    # One entry per timestep for every parameter (static parameters are broadcast).
    F, B, b, Q, H, D, d, R = vmap(lambda t: _get_params(params, num_timesteps, t))(jnp.arange(num_timesteps))
    if R.ndim == 2:   # diagonal emission covariance given as (T, N)
        R = vmap(jnp.diag)(R)

    # Emission terms, every timestep.
    R_inv, R_logdet = vmap(_inverse_and_logdet)(R)
    residuals = emissions - jnp.einsum("tnm,tm->tn", D, inputs) - d
    J_diag = jnp.einsum("tnd,tne,tef->tdf", H, R_inv, H)
    h = jnp.einsum("tnd,tne,te->td", H, R_inv, residuals)
    log_constant = -0.5 * (num_timesteps * emission_dim * jnp.log(2 * jnp.pi) + R_logdet.sum()
                           + jnp.einsum("tn,tnm,tm->", residuals, R_inv, residuals))

    # Initial state.
    S0_inv, S0_logdet = _inverse_and_logdet(params.initial.cov)
    m0 = params.initial.mean
    J_diag = J_diag.at[0].add(S0_inv)
    h = h.at[0].add(S0_inv @ m0)
    log_constant -= 0.5 * (state_dim * jnp.log(2 * jnp.pi) + S0_logdet + m0 @ S0_inv @ m0)

    # Dynamics: the transition into x_t uses entry t of F, B, b, Q and u_t, for t >= 1. Entry 0
    # is sliced away, never masked, so a NaN there cannot leak into the result.
    F, B, b, Q = F[1:], B[1:], b[1:], Q[1:]
    Q_inv, Q_logdet = vmap(_inverse_and_logdet)(Q)
    c = jnp.einsum("tdm,tm->td", B, inputs[1:]) + b
    J_diag = J_diag.at[:-1].add(jnp.einsum("tji,tjk,tkl->til", F, Q_inv, F))   # F^T Q^{-1} F
    J_diag = J_diag.at[1:].add(Q_inv)
    J_lower = -jnp.einsum("tij,tjk->tik", Q_inv, F)
    h = h.at[:-1].add(-jnp.einsum("tji,tjk,tk->ti", F, Q_inv, c))             # -F^T Q^{-1} c
    h = h.at[1:].add(jnp.einsum("tij,tj->ti", Q_inv, c))
    log_constant -= 0.5 * ((num_timesteps - 1) * state_dim * jnp.log(2 * jnp.pi) + Q_logdet.sum()
                           + jnp.einsum("td,tde,te->", c, Q_inv, c))

    return BlockTridiagPotentials(J_diag=symmetrize(J_diag), J_lower=J_lower, h=h), log_constant
