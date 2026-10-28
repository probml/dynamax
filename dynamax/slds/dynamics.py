r"""
Linear Gaussian dynamics component of a switching linear dynamical system.

.. math::
    x_0 \sim \mathcal{N}(m_0, \Sigma_0), \qquad
    x_t \mid x_{t-1}, z_t = k, u_t \sim \mathcal{N}(F_k x_{t-1} + b_k + B_k u_t, Q_k) \quad (t \geq 1)

The initial distribution is shared across states and frozen at :math:`(0, I)` by default, as
in ``ssm``. The M-step reproduces ``ssm``'s regularized least squares (ridge toward
:math:`F_k = I`, :math:`B_k = 0`, :math:`b_k = 0` and an inverse-Wishart term on
:math:`Q_k`) with the same default hyperparameters.
"""
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, Float

from dynamax.parameters import ParameterProperties, ensure_all_or_none_trainable
from dynamax.slds.abstractions import DynamicsStats, SLDSDynamics
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import psd_solve, random_rotation, symmetrize


class ParamsSLDSDynamics(NamedTuple):
    r"""Parameters of the linear Gaussian SLDS dynamics (``ssm``: ``mu_init``, ``Sigmas_init``,
    ``As``, ``bs``, ``Vs``, ``Sigmas``).

    :param initial_mean: :math:`m_0`, shared across states
    :param initial_cov: :math:`\Sigma_0`, shared across states
    :param weights: :math:`F_k`
    :param biases: :math:`b_k`
    :param input_weights: :math:`B_k`
    :param covs: :math:`Q_k`, full (K, D, D) or diagonal (K, D)
    """
    initial_mean: Union[Float[Array, " state_dim"], ParameterProperties]
    initial_cov: Union[Float[Array, "state_dim state_dim"], ParameterProperties]
    weights: Union[Float[Array, "num_states state_dim state_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_states state_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "num_states state_dim state_dim"], Float[Array, "num_states state_dim"], ParameterProperties]


def _full_cov(cov: Array) -> Array:
    """Expand a batch of diagonal covariances (K, D) to (K, D, D); full covariances (K, D, D) pass through."""
    return cov if cov.ndim == 3 else vmap(jnp.diag)(cov)


def _gaussian_log_prob(residual: Array, cov: Array) -> Array:
    """log N(residual | 0, cov) for a batch of residuals (..., D) and one covariance (D, D)."""
    dim = residual.shape[-1]
    chol = jnp.linalg.cholesky(symmetrize(cov))
    whitened = jnp.linalg.solve(chol, residual[..., None])[..., 0]
    log_det = 2 * jnp.log(jnp.diag(chol)).sum()
    return -0.5 * (dim * jnp.log(2 * jnp.pi) + log_det + jnp.sum(whitened ** 2, axis=-1))


def _cov_inverse_and_logdet(cov: Array) -> Tuple[Array, Array]:
    """Inverse and log determinant of one (D, D) covariance."""
    chol = jnp.linalg.cholesky(symmetrize(cov))
    inv = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, jnp.eye(cov.shape[-1], dtype=cov.dtype)))
    return symmetrize(inv), 2 * jnp.log(jnp.diag(chol)).sum()


class LinearGaussianSLDSDynamics(SLDSDynamics):
    r"""Linear Gaussian dynamics with one linear map per discrete state.

    :param num_states: :math:`K`
    :param state_dim: :math:`D`
    :param input_dim: :math:`M` (may be 0)
    :param cov_type: ``"full"`` or ``"diag"`` noise covariance :math:`Q_k`
    :param l2_penalty_weights: ridge toward :math:`F_k = I` (``ssm`` ``l2_penalty_A``)
    :param l2_penalty_input_weights: ridge toward :math:`B_k = 0` (``ssm`` ``l2_penalty_V``)
    :param l2_penalty_bias: ridge toward :math:`b_k = 0` (``ssm`` ``l2_penalty_b``)
    :param nu0: inverse-Wishart degrees of freedom; the MLE is ``nu0 = -(state_dim + 1)``
    :param Psi0: inverse-Wishart scale, :math:`\Psi_0 I`
    """
    is_quadratic_in_x = True

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 input_dim: int = 0,
                 cov_type: str = "full",
                 l2_penalty_weights: float = 1e-8,
                 l2_penalty_input_weights: float = 1e-8,
                 l2_penalty_bias: float = 1e-8,
                 nu0: float = 1e-4,
                 Psi0: float = 1e-4):
        if cov_type not in ("full", "diag"):
            raise ValueError(f"cov_type must be 'full' or 'diag', got {cov_type!r}")
        self.num_states = num_states
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.cov_type = cov_type
        self.l2_penalty_weights = l2_penalty_weights
        self.l2_penalty_input_weights = l2_penalty_input_weights
        self.l2_penalty_bias = l2_penalty_bias
        self.nu0 = nu0
        self.Psi0 = Psi0

    # ---- parameters -------------------------------------------------------------------------
    @property
    def num_regressors(self) -> int:
        """P = D + M + 1: the regressor is [x_{t-1}; u_t; 1]."""
        return self.state_dim + self.input_dim + 1

    def _ridge(self) -> Tuple[Array, Array, Array]:
        """J0 (P, P), h0 (P, D) and the prior mean Theta0 = [I 0 0] (D, P) of the regression."""
        D, M = self.state_dim, self.input_dim
        J0 = jnp.diag(jnp.concatenate([self.l2_penalty_weights * jnp.ones(D),
                                       self.l2_penalty_input_weights * jnp.ones(M),
                                       self.l2_penalty_bias * jnp.ones(1)]))
        theta0 = jnp.concatenate([jnp.eye(D), jnp.zeros((D, M + 1))], axis=1)
        return J0, J0 @ theta0.T, theta0

    def initialize(self,
                   key: Optional[PRNGKeyT] = None,
                   method: str = "prior",
                   initial_mean: Optional[Array] = None,
                   initial_cov: Optional[Array] = None,
                   weights: Optional[Array] = None,
                   biases: Optional[Array] = None,
                   input_weights: Optional[Array] = None,
                   covs: Optional[Array] = None
                   ) -> Tuple[ParamsSLDSDynamics, ParamsSLDSDynamics]:
        """Random (``method="prior"``) or user-supplied parameters and their properties.

        ``"prior"`` sets each :math:`F_k` to 0.95 times a random rotation, biases and input
        weights to zero and noise covariances to the identity. The initial distribution is
        :math:`(0, I)` and frozen, as in ``ssm``.
        """
        K, D, M = self.num_states, self.state_dim, self.input_dim
        if weights is None:
            if key is None:
                raise ValueError("key must be provided if weights are not provided.")
            weights = vmap(lambda k: 0.95 * random_rotation(k, D))(jr.split(key, K))
        default_cov = jnp.ones((K, D)) if self.cov_type == "diag" else jnp.tile(jnp.eye(D), (K, 1, 1))
        if covs is not None and covs.shape != default_cov.shape:
            raise ValueError(f"covs must have shape {default_cov.shape} for cov_type={self.cov_type!r}, got {covs.shape}.")
        params = ParamsSLDSDynamics(
            initial_mean=jnp.zeros(D) if initial_mean is None else initial_mean,
            initial_cov=jnp.eye(D) if initial_cov is None else initial_cov,
            weights=weights,
            biases=jnp.zeros((K, D)) if biases is None else biases,
            input_weights=jnp.zeros((K, D, M)) if input_weights is None else input_weights,
            covs=default_cov if covs is None else covs)
        cov_constrainer = tfb.Exp() if self.cov_type == "diag" else RealToPSDBijector()
        props = ParamsSLDSDynamics(
            initial_mean=ParameterProperties(trainable=False),
            initial_cov=ParameterProperties(trainable=False, constrainer=RealToPSDBijector()),
            weights=ParameterProperties(),
            biases=ParameterProperties(),
            input_weights=ParameterProperties(),
            covs=ParameterProperties(constrainer=cov_constrainer))
        return params, props

    def log_prior(self, params: ParamsSLDSDynamics) -> Scalar:
        r"""Unnormalized log prior whose maximizer is the M-step below:
        :math:`\sum_k [-\tfrac12 \mathrm{tr}(Q_k^{-1}((\Theta_k - \Theta_0) J_0 (\Theta_k - \Theta_0)^\top + \Psi_0 I))
        - \tfrac12 (\nu_0 + D + 1) \log|Q_k|]`. Zero in the MLE configuration."""
        J0, _, theta0 = self._ridge()
        D = self.state_dim

        def _one(F, B, b, Q):
            theta = jnp.concatenate([F, B, b[:, None]], axis=1) - theta0
            Q_inv, Q_logdet = _cov_inverse_and_logdet(Q)
            return -0.5 * jnp.trace(Q_inv @ (theta @ J0 @ theta.T + self.Psi0 * jnp.eye(D))) \
                   - 0.5 * (self.nu0 + D + 1) * Q_logdet

        return vmap(_one)(params.weights, params.input_weights, params.biases, _full_cov(params.covs)).sum()

    # ---- likelihood terms -------------------------------------------------------------------
    def _predicted_means(self, params: ParamsSLDSDynamics, x_prev: Array, inputs: Array) -> Array:
        """F_k x_prev + b_k + B_k u for every state: (T', K, D)."""
        return (jnp.einsum("kij,tj->tki", params.weights, x_prev)
                + jnp.einsum("kim,tm->tki", params.input_weights, inputs)
                + params.biases[None])

    def ar_log_likelihoods(self, params: ParamsSLDSDynamics, x_prev: Array, x: Array, inputs: Array) -> Array:
        """log N(x_t | F_k x_prev_t + b_k + B_k u_t, Q_k) for every row t and state k: (T', K)."""
        residuals = x[:, None, :] - self._predicted_means(params, x_prev, inputs)   # (T', K, D)
        covs = _full_cov(params.covs)
        return vmap(lambda r, Q: _gaussian_log_prob(r, Q), in_axes=(1, 0), out_axes=1)(residuals, covs)

    def log_likelihoods(self, params: ParamsSLDSDynamics, x: Array, inputs: Array) -> Array:
        """(T, K): row 0 is log N(x_0 | m_0, Sigma_0) for every state, rows t >= 1 the AR terms."""
        row0 = _gaussian_log_prob(x[0] - params.initial_mean, params.initial_cov)
        rest = self.ar_log_likelihoods(params, x[:-1], x[1:], inputs[1:])
        return jnp.concatenate([jnp.full((1, self.num_states), row0), rest], axis=0)

    def expected_log_likelihoods(self, params: ParamsSLDSDynamics, qx, inputs: Array) -> Array:
        r"""Exact :math:`\mathbb{E}_{q(x)}[\log p(x_t \mid x_{t-1}, z_t = k, u_t)]`, shape (T, K).

        With :math:`r_t = \mu_t - F_k \mu_{t-1} - c_t` and the centered cross-covariance
        :math:`C_t = \mathbb{E}[x_{t-1} x_t^\top] - \mu_{t-1} \mu_t^\top`, the expected squared
        residual is :math:`r_t r_t^\top + \Sigma_t - F_k C_t - C_t^\top F_k^\top + F_k \Sigma_{t-1} F_k^\top`.
        """
        means, covs, crosses = qx.smoothed_means, qx.smoothed_covariances, qx.smoothed_cross_covariances
        D = self.state_dim
        # Row 0: the shared initial distribution.
        S0_inv, S0_logdet = _cov_inverse_and_logdet(params.initial_cov)
        r0 = means[0] - params.initial_mean
        row0 = -0.5 * (D * jnp.log(2 * jnp.pi) + S0_logdet + jnp.trace(S0_inv @ (covs[0] + jnp.outer(r0, r0))))
        # Rows t >= 1.
        centered_cross = crosses - jnp.einsum("ti,tj->tij", means[:-1], means[1:])
        residuals = means[1:, None, :] - self._predicted_means(params, means[:-1], inputs[1:])   # (T-1, K, D)
        Q_full = _full_cov(params.covs)
        Q_inv, Q_logdet = vmap(_cov_inverse_and_logdet)(Q_full)

        def _one_state(F, Q_inv_k, Q_logdet_k, res_k):
            """(T-1,) expected log-likelihoods for one state."""
            M = (jnp.einsum("ti,tj->tij", res_k, res_k) + covs[1:]
                 - jnp.einsum("ij,tjk->tik", F, centered_cross)
                 - jnp.einsum("tji,kj->tik", centered_cross, F)
                 + jnp.einsum("ij,tjk,lk->til", F, covs[:-1], F))
            return -0.5 * (D * jnp.log(2 * jnp.pi) + Q_logdet_k + jnp.einsum("ij,tji->t", Q_inv_k, M))

        rest = vmap(_one_state, in_axes=(0, 0, 0, 1), out_axes=1)(params.weights, Q_inv, Q_logdet, residuals)
        return jnp.concatenate([jnp.full((1, self.num_states), row0), rest], axis=0)

    def expected_log_joint_x(self, params: ParamsSLDSDynamics, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """sum_t sum_k Ez[t, k] log p(x_t | x_{t-1}, z_t = k, u_t), evaluated at the point x."""
        return jnp.sum(smoothed_probs * self.log_likelihoods(params, x, inputs))

    def neg_hessian_blocks(self, params: ParamsSLDSDynamics, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        r"""Block-tridiagonal negative Hessian of :math:`\sum_t \sum_k \mathbb{E}[z_t = k] \log p(x_t \mid x_{t-1}, z_t = k)`
        in :math:`x`, which does not depend on :math:`x`: ``(J_diag (T, D, D), J_lower (T-1, D, D))``."""
        num_timesteps = smoothed_probs.shape[0]
        Q_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        FQF = jnp.einsum("kji,kjl,klm->kim", params.weights, Q_inv, params.weights)   # F^T Q^{-1} F
        QF = jnp.einsum("kij,kjl->kil", Q_inv, params.weights)                         # Q^{-1} F
        w_next = smoothed_probs[1:]                                                    # Ez[t+1]
        S0_inv, _ = _cov_inverse_and_logdet(params.initial_cov)
        J_diag = jnp.zeros((num_timesteps, self.state_dim, self.state_dim)).at[0].set(S0_inv)
        J_diag = J_diag.at[:-1].add(jnp.einsum("tk,kij->tij", w_next, FQF))
        J_diag = J_diag.at[1:].add(jnp.einsum("tk,kij->tij", w_next, Q_inv))
        J_lower = -jnp.einsum("tk,kij->tij", w_next, QF)
        return J_diag, J_lower

    def sample_transition(self, key: PRNGKeyT, params: ParamsSLDSDynamics, state: Array, x_prev: Array, inpt: Array) -> Array:
        """One draw of x_t given z_t = state, x_{t-1} = x_prev and u_t = inpt."""
        mean = params.weights[state] @ x_prev + params.input_weights[state] @ inpt + params.biases[state]
        cov = _full_cov(params.covs)[state]
        return mean + jnp.linalg.cholesky(symmetrize(cov)) @ jr.normal(key, (self.state_dim,))

    # ---- sufficient statistics --------------------------------------------------------------
    def _regression_stats(self, weights, x_prev, x, inputs, cov_prev=None, centered_cross=None, cov=None):
        """Weighted second moments of phi = [x_prev; u; 1] and x over all rows, per state."""
        num_rows = x.shape[0]
        phi_mean = jnp.concatenate([x_prev, inputs, jnp.ones((num_rows, 1))], axis=1)          # (T', P)
        phiphiT = jnp.einsum("ti,tj->tij", phi_mean, phi_mean)
        phixT = jnp.einsum("ti,tj->tij", phi_mean, x)
        xxT = jnp.einsum("ti,tj->tij", x, x)
        if cov_prev is not None:
            D = self.state_dim
            phiphiT = phiphiT.at[:, :D, :D].add(cov_prev)
            phixT = phixT.at[:, :D, :].add(centered_cross)
            xxT = xxT + cov
        return DynamicsStats(
            Ex0=jnp.zeros(self.state_dim), Ex0x0T=jnp.zeros((self.state_dim, self.state_dim)),
            sum_w=weights.sum(0),
            sum_phiphiT=jnp.einsum("tk,tij->kij", weights, phiphiT),
            sum_phixT=jnp.einsum("tk,tij->kij", weights, phixT),
            sum_xxT=jnp.einsum("tk,tij->kij", weights, xxT))

    def collect_suff_stats(self, params: ParamsSLDSDynamics, smoothed_probs: Array, qx, inputs: Array) -> DynamicsStats:
        """Exact expected statistics under q(x): pairs t >= 1 weighted by Ez[t, k], plus the moments of x_0."""
        means, covs, crosses = qx.smoothed_means, qx.smoothed_covariances, qx.smoothed_cross_covariances
        centered_cross = crosses - jnp.einsum("ti,tj->tij", means[:-1], means[1:])
        stats = self._regression_stats(smoothed_probs[1:], means[:-1], means[1:], inputs[1:], covs[:-1], centered_cross, covs[1:])
        return stats._replace(Ex0=means[0], Ex0x0T=covs[0] + jnp.outer(means[0], means[0]))

    # ---- M-step -----------------------------------------------------------------------------
    def initialize_m_step_state(self, params: ParamsSLDSDynamics, props: ParamsSLDSDynamics) -> Any:
        """The closed-form M-step needs no state."""
        return None

    def m_step(self,
               params: ParamsSLDSDynamics,
               props: ParamsSLDSDynamics,
               batch_stats: DynamicsStats,
               m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParamsSLDSDynamics, Any]:
        r"""``ssm``'s regularized least squares per state, plus the initial-state moments.

        :math:`\Theta_k = (S_{\phi x} + h_0)^\top (S_{\phi\phi} + J_0)^{-1}` and
        :math:`Q_k = (\mathrm{sqerr}_k + \Psi_0 I) / (\nu_0 + n_k + D + 1)`, where ``sqerr`` includes
        the ridge penalty so that the update maximizes ``log_prior`` plus the expected log
        likelihood exactly. States with fewer than one expected observation are replaced by a
        perturbed copy of a random used state when ``key`` is given, as in ``ssm``.
        """
        stats = tree_map(lambda x: jnp.sum(x, axis=0), batch_stats)
        num_sequences = batch_stats.Ex0.shape[0]
        D, M = self.state_dim, self.input_dim

        if ensure_all_or_none_trainable((props.weights, props.biases, props.input_weights, props.covs),
                                        what="dynamics weights, biases, input weights and covariances"):
            J0, h0, theta0 = self._ridge()

            def _one_state(S_phiphi, S_phix, S_xx, n):
                theta = psd_solve(S_phiphi + J0, S_phix + h0).T                      # (D, P)
                dtheta = theta - theta0
                sqerr = (S_xx - theta @ S_phix - S_phix.T @ theta.T + theta @ S_phiphi @ theta.T
                         + dtheta @ J0 @ dtheta.T)
                Q = symmetrize((sqerr + self.Psi0 * jnp.eye(D)) / (self.nu0 + n + D + 1))
                return theta[:, :D], theta[:, D:D + M], theta[:, -1], Q

            F, B, b, Q = vmap(_one_state)(stats.sum_phiphiT, stats.sum_phixT, stats.sum_xxT, stats.sum_w)

            if key is not None:
                F, B, b, Q = self._perturb_unused_states(key, stats.sum_w, F, B, b, Q)
            if self.cov_type == "diag":
                Q = vmap(jnp.diag)(Q)
            params = params._replace(weights=F, input_weights=B, biases=b, covs=Q)

        if ensure_all_or_none_trainable((props.initial_mean, props.initial_cov), what="initial mean and covariance"):
            m = stats.Ex0 / num_sequences
            S = symmetrize(stats.Ex0x0T / num_sequences - jnp.outer(m, m))
            params = params._replace(initial_mean=m, initial_cov=S)
        return params, m_step_state

    def _perturb_unused_states(self, key, sum_w, F, B, b, Q):
        """ssm's rule: a state with sum_w < 1 copies a random state with sum_w > 1, plus 0.01 N(0, I) noise."""
        K = self.num_states
        unused, used = sum_w < 1, sum_w > 1
        any_used = jnp.any(used)
        probs = jnp.where(any_used, used / jnp.maximum(used.sum(), 1), jnp.ones(K) / K)
        k_pick, kF, kB, kb = jr.split(key, 4)
        donors = vmap(lambda k: jr.choice(k, K, p=probs))(jr.split(k_pick, K))
        replace = unused & any_used
        F = jnp.where(replace[:, None, None], F[donors] + 0.01 * jr.normal(kF, F.shape), F)
        B = jnp.where(replace[:, None, None], B[donors] + 0.01 * jr.normal(kB, B.shape), B)
        b = jnp.where(replace[:, None], b[donors] + 0.01 * jr.normal(kb, b.shape), b)
        Q = jnp.where(replace[:, None, None], Q[donors], Q)
        return F, B, b, Q
