r"""
Linear Gaussian emissions of a switching linear dynamical system.

.. math::
    y_t \mid x_t, z_t = k, u_t \sim \mathcal{N}(H_k x_t + d_k + D_k u_t, R_k)

With ``single_subspace=True`` (the default) all states share one emission model (``Keff = 1``);
otherwise each state has its own (``Keff = K``). The noise covariance is diagonal by default
(``ssm``'s ``inv_etas``). The M-step reproduces ``ssm``'s ``fit_linear_regression`` with its
default ridge and inverse-Wishart regularization.
"""
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, Float

from dynamax.parameters import ParameterProperties, ensure_all_or_none_trainable
from dynamax.slds.abstractions import EmissionStats, SLDSEmissions
from dynamax.slds.dynamics import _cov_inverse_and_logdet, _full_cov, _gaussian_log_prob
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import psd_solve, symmetrize


class ParamsSLDSEmissions(NamedTuple):
    r"""Parameters of the linear Gaussian SLDS emissions (``ssm``: ``Cs``, ``ds``, ``Fs``, ``inv_etas``).

    :param weights: :math:`H_k`
    :param biases: :math:`d_k`
    :param input_weights: :math:`D_k`
    :param covs: :math:`R_k`, diagonal (Keff, N) or full (Keff, N, N)
    """
    weights: Union[Float[Array, "num_subspaces emission_dim state_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_subspaces emission_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_subspaces emission_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "num_subspaces emission_dim emission_dim"], Float[Array, "num_subspaces emission_dim"], ParameterProperties]


class LinearGaussianSLDSEmissions(SLDSEmissions):
    r"""Linear Gaussian emissions, shared across states or one per state.

    :param num_states: :math:`K`
    :param state_dim: :math:`D`
    :param emission_dim: :math:`N`
    :param input_dim: :math:`M` (may be 0)
    :param single_subspace: share one emission model across states
    :param cov_type: ``"diag"`` (default, as ``ssm``) or ``"full"`` noise covariance
    :param l2_penalty: ridge toward zero on :math:`[H_k, D_k, d_k]` (``ssm`` ``prior_ExxT = 1e-4 I``)
    :param nu0: inverse-Wishart degrees of freedom; the MLE is ``nu0 = -(emission_dim + 1)``
    :param Psi0: inverse-Wishart scale, :math:`\Psi_0 I`
    """
    is_quadratic_in_x = True

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 emission_dim: int,
                 input_dim: int = 0,
                 single_subspace: bool = True,
                 cov_type: str = "diag",
                 l2_penalty: float = 1e-4,
                 nu0: float = 1.0,
                 Psi0: float = 1.0):
        if cov_type not in ("full", "diag"):
            raise ValueError(f"cov_type must be 'full' or 'diag', got {cov_type!r}")
        self.num_states = num_states
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.single_subspace = single_subspace
        self.cov_type = cov_type
        self.l2_penalty = l2_penalty
        self.nu0 = nu0
        self.Psi0 = Psi0

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    @property
    def num_subspaces(self) -> int:
        """Keff: 1 with a single subspace, else K."""
        return 1 if self.single_subspace else self.num_states

    @property
    def num_regressors(self) -> int:
        """P = D + M + 1: the regressor is [x_t; u_t; 1]."""
        return self.state_dim + self.input_dim + 1

    def state_weights(self, smoothed_probs: Array) -> Array:
        """(T, Keff) weights of the per-subspace terms: Ez, or ones with a single subspace."""
        return jnp.ones((smoothed_probs.shape[0], 1)) if self.single_subspace else smoothed_probs

    # ---- parameters -------------------------------------------------------------------------
    def initialize(self,
                   key: Optional[PRNGKeyT] = None,
                   method: str = "prior",
                   weights: Optional[Array] = None,
                   biases: Optional[Array] = None,
                   input_weights: Optional[Array] = None,
                   covs: Optional[Array] = None
                   ) -> Tuple[ParamsSLDSEmissions, ParamsSLDSEmissions]:
        """Random (``method="prior"``: standard normal weights, zero biases, unit covariances) or
        user-supplied parameters and their properties."""
        Keff, N, D, M = self.num_subspaces, self.emission_dim, self.state_dim, self.input_dim
        if weights is None:
            if key is None:
                raise ValueError("key must be provided if weights are not provided.")
            weights = jr.normal(key, (Keff, N, D))
        default_cov = jnp.ones((Keff, N)) if self.cov_type == "diag" else jnp.tile(jnp.eye(N), (Keff, 1, 1))
        if covs is not None and covs.shape != default_cov.shape:
            raise ValueError(f"covs must have shape {default_cov.shape} for cov_type={self.cov_type!r}, got {covs.shape}.")
        params = ParamsSLDSEmissions(
            weights=weights,
            biases=jnp.zeros((Keff, N)) if biases is None else biases,
            input_weights=jnp.zeros((Keff, N, M)) if input_weights is None else input_weights,
            covs=default_cov if covs is None else covs)
        cov_constrainer = tfb.Exp() if self.cov_type == "diag" else RealToPSDBijector()
        props = ParamsSLDSEmissions(weights=ParameterProperties(), biases=ParameterProperties(),
                                    input_weights=ParameterProperties(),
                                    covs=ParameterProperties(constrainer=cov_constrainer))
        return params, props

    def log_prior(self, params: ParamsSLDSEmissions) -> Scalar:
        r"""Unnormalized log prior whose maximizer is the M-step below:
        :math:`\sum_k [-\tfrac12 \mathrm{tr}(R_k^{-1}(\lambda \Theta_k \Theta_k^\top + \Psi_0 I)) - \tfrac12 (\nu_0 + N + 1) \log|R_k|]`."""
        N = self.emission_dim

        def _one(H, D, d, R):
            theta = jnp.concatenate([H, D, d[:, None]], axis=1)
            R_inv, R_logdet = _cov_inverse_and_logdet(R)
            return -0.5 * jnp.trace(R_inv @ (self.l2_penalty * theta @ theta.T + self.Psi0 * jnp.eye(N))) \
                   - 0.5 * (self.nu0 + N + 1) * R_logdet

        return vmap(_one)(params.weights, params.input_weights, params.biases, _full_cov(params.covs)).sum()

    # ---- likelihood terms -------------------------------------------------------------------
    def _predicted(self, params: ParamsSLDSEmissions, x: Array, inputs: Array) -> Array:
        """H_k x_t + d_k + D_k u_t for every subspace: (T, Keff, N)."""
        return (jnp.einsum("knd,td->tkn", params.weights, x)
                + jnp.einsum("knm,tm->tkn", params.input_weights, inputs)
                + params.biases[None])

    def log_likelihoods(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, inputs: Array, mask=None) -> Array:
        """(T, Keff) log N(y_t | H_k x_t + d_k + D_k u_t, R_k)."""
        assert mask is None, "emission masks are not supported yet"
        residuals = emissions[:, None, :] - self._predicted(params, x, inputs)
        return vmap(lambda r, R: _gaussian_log_prob(r, R), in_axes=(1, 0), out_axes=1)(residuals, _full_cov(params.covs))

    def expected_log_likelihoods(self, params: ParamsSLDSEmissions, emissions: Array, qx, inputs: Array, mask=None) -> Array:
        """(T, Keff): the plug-in log likelihood at the posterior mean minus ½ tr(R_k⁻¹ H_k Σ_t H_kᵀ)."""
        plug_in = self.log_likelihoods(params, emissions, qx.smoothed_means, inputs, mask)
        R_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        correction = 0.5 * jnp.einsum("knd,tde,kne->tk",
                                      jnp.einsum("kij,kjd->kid", R_inv, params.weights), qx.smoothed_covariances, params.weights)
        return plug_in - correction

    def expected_log_joint_x(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Scalar:
        """sum_t sum_k w[t, k] log p(y_t | x_t, z_t = k, u_t) at the point x."""
        return jnp.sum(self.state_weights(smoothed_probs) * self.log_likelihoods(params, emissions, x, inputs, mask))

    def neg_hessian_blocks(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Array:
        """(T, D, D): sum_k w[t, k] H_kᵀ R_k⁻¹ H_k, independent of x."""
        R_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        HRH = jnp.einsum("knd,knm,kme->kde", params.weights, R_inv, params.weights)
        return jnp.einsum("tk,kde->tde", self.state_weights(smoothed_probs), HRH)

    def invert(self, params: ParamsSLDSEmissions, emissions: Array, inputs: Array) -> Array:
        """Least-squares inversion with the subspace-averaged parameters, x̂_t = H̄⁺ (y_t − d̄ − D̄ u_t)."""
        H, D, d = params.weights.mean(0), params.input_weights.mean(0), params.biases.mean(0)
        residuals = emissions - inputs @ D.T - d
        return residuals @ jnp.linalg.pinv(H).T

    def expected_emissions(self, params: ParamsSLDSEmissions, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """(T, N): sum_k w[t, k] (H_k mean_t + d_k + D_k u_t)."""
        return jnp.einsum("tk,tkn->tn", self.state_weights(smoothed_probs), self._predicted(params, means, inputs))

    def sample_emission(self, key: PRNGKeyT, params: ParamsSLDSEmissions, state: Array, x: Array, inpt: Array) -> Array:
        """One draw of y_t given z_t = state, x_t = x and u_t = inpt."""
        k = 0 if self.single_subspace else state
        mean = params.weights[k] @ x + params.input_weights[k] @ inpt + params.biases[k]
        cov = _full_cov(params.covs)[k]
        return mean + jnp.linalg.cholesky(symmetrize(cov)) @ jr.normal(key, (self.emission_dim,))

    # ---- sufficient statistics and M-step ---------------------------------------------------
    def collect_suff_stats(self, params: ParamsSLDSEmissions, emissions: Array, smoothed_probs: Array, qx, inputs: Array, mask=None) -> EmissionStats:
        """Expected statistics of the regression of y_t on psi_t = [x_t; u_t; 1] under q(x)."""
        assert mask is None, "emission masks are not supported yet"
        num_timesteps, D = qx.smoothed_means.shape
        psi = jnp.concatenate([qx.smoothed_means, inputs, jnp.ones((num_timesteps, 1))], axis=1)
        psipsiT = jnp.einsum("ti,tj->tij", psi, psi).at[:, :D, :D].add(qx.smoothed_covariances)
        psiyT = jnp.einsum("ti,tj->tij", psi, emissions)
        yyT = jnp.einsum("ti,tj->tij", emissions, emissions)
        w = self.state_weights(smoothed_probs)
        return EmissionStats(sum_w=w.sum(0),
                             sum_psipsiT=jnp.einsum("tk,tij->kij", w, psipsiT),
                             sum_psiyT=jnp.einsum("tk,tij->kij", w, psiyT),
                             sum_yyT=jnp.einsum("tk,tij->kij", w, yyT))

    def initialize_m_step_state(self, params: ParamsSLDSEmissions, props: ParamsSLDSEmissions) -> Any:
        """The closed-form M-step needs no state."""
        return None

    def m_step(self, params: ParamsSLDSEmissions, props: ParamsSLDSEmissions, batch_stats: EmissionStats, m_step_state: Any
               ) -> Tuple[ParamsSLDSEmissions, Any]:
        r"""``ssm``'s ``fit_linear_regression``: :math:`\Theta_k = S_{\psi y}^\top (S_{\psi\psi} + \lambda I)^{-1}` and
        :math:`R_k = (\mathrm{sqerr}_k + \Psi_0 I) / (\nu_0 + n_k + N + 1)`, where ``sqerr`` is formed with the
        ridged second moment (as ``ssm``) so that the update maximizes ``log_prior`` plus the expected log likelihood."""
        if not ensure_all_or_none_trainable(props, what="emission parameters"):
            return params, m_step_state
        stats = tree_map(lambda x: jnp.sum(x, axis=0), batch_stats)
        N, D, M, P = self.emission_dim, self.state_dim, self.input_dim, self.num_regressors

        def _one(S_psipsi, S_psiy, S_yy, n):
            S_ridged = S_psipsi + self.l2_penalty * jnp.eye(P)
            theta = psd_solve(S_ridged, S_psiy).T                                   # (N, P)
            sqerr = S_yy - theta @ S_psiy - S_psiy.T @ theta.T + theta @ S_ridged @ theta.T
            R = symmetrize((sqerr + self.Psi0 * jnp.eye(N)) / (self.nu0 + n + N + 1))
            return theta[:, :D], theta[:, D:D + M], theta[:, -1], R

        H, Dm, d, R = vmap(_one)(stats.sum_psipsiT, stats.sum_psiyT, stats.sum_yyT, stats.sum_w)
        if self.cov_type == "diag":
            R = vmap(jnp.diag)(R)
        return params._replace(weights=H, input_weights=Dm, biases=d, covs=R), m_step_state
