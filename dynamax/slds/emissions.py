"""
Linear Gaussian emissions of a switching linear dynamical system.

Each emission is a linear function of the continuous state and the inputs, plus Gaussian noise.
By default all discrete states share one emission model; otherwise each discrete state has its own.
The noise covariance is diagonal by default.
"""
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Array, Float

from dynamax.parameters import ParameterProperties, ensure_all_or_none_trainable
from dynamax.slds.abstractions import EmissionStats, SLDSEmissions
from dynamax.slds.dynamics import _cov_inverse_and_logdet, _full_cov
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import psd_solve, symmetrize


class ParamsSLDSEmissions(NamedTuple):
    """Parameters of the linear Gaussian emissions of an SLDS, one set per discrete state or one shared set.

    :param weights: emission weights, shape (K, N, D), or (1, N, D) with a single subspace
    :param biases: emission biases, shape (K, N) or (1, N)
    :param input_weights: emission input weights, shape (K, N, M) or (1, N, M)
    :param covs: emission covariances, shape (K, N) when diagonal or (K, N, N) when full, with 1 in
        place of K for a single subspace
    """
    weights: Union[Float[Array, "num_subspaces emission_dim state_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_subspaces emission_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_subspaces emission_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "num_subspaces emission_dim emission_dim"], Float[Array, "num_subspaces emission_dim"], ParameterProperties]


class LinearGaussianSLDSEmissions(SLDSEmissions):
    """Linear Gaussian emissions, shared across discrete states or one set per state.

    :param num_states: number of discrete states K
    :param state_dim: dimension of the continuous state D
    :param emission_dim: dimension of the emissions N
    :param input_dim: dimension of the inputs M (may be 0)
    :param single_subspace: whether all discrete states share one emission model
    :param cov_type: ``"diag"`` (default) or ``"full"`` noise covariance
    :param l2_penalty: strength of the penalty that shrinks the weights, input weights and biases toward zero
    :param nu0: degrees of freedom of the inverse-Wishart prior on the noise covariance; with
        ``nu0 = -(emission_dim + 1)``, ``Psi0 = 0`` and ``l2_penalty = 0`` the M-step is the maximum-likelihood estimate
    :param Psi0: scale of the inverse-Wishart prior, a multiple of the identity
    """
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
        """Number of emission models, Keff: 1 with a single subspace, otherwise the number of discrete states."""
        return 1 if self.single_subspace else self.num_states

    @property
    def num_regressors(self) -> int:
        """Number of regressors, D + M + 1: the continuous state, the inputs and a constant."""
        return self.state_dim + self.input_dim + 1

    def state_weights(self, smoothed_probs: Array) -> Array:
        """Weight of each emission model at each time step, shape (T, Keff)."""
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
        """Initialize the emission parameters and their properties.

        Defaults: standard normal weights (needs ``key``), zero biases and input weights, and identity covariances.
        Arguments passed in replace them. ``method`` is not used.
        """
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
        """Compute the log prior of the emission parameters, up to a constant: the penalty on the weights, input
        weights and biases plus the inverse-Wishart prior on the covariances."""
        N = self.emission_dim

        def _one(H, Dm, d, R):
            theta = jnp.concatenate([H, Dm, d[:, None]], axis=1)
            R_inv, R_logdet = _cov_inverse_and_logdet(R)
            return -0.5 * jnp.trace(R_inv @ (self.l2_penalty * theta @ theta.T + self.Psi0 * jnp.eye(N))) \
                   - 0.5 * (self.nu0 + N + 1) * R_logdet

        return vmap(_one)(params.weights, params.input_weights, params.biases, _full_cov(params.covs)).sum()

    # ---- likelihood terms -------------------------------------------------------------------
    def _predicted(self, params: ParamsSLDSEmissions, x: Array, inputs: Array) -> Array:
        """Mean emission under each emission model at each time step, shape (T, Keff, N)."""
        return (jnp.einsum("knd,td->tkn", params.weights, x)
                + jnp.einsum("knm,tm->tkn", params.input_weights, inputs)
                + params.biases[None])

    def log_likelihoods(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, inputs: Array) -> Array:
        """Compute the log likelihood of each emission under each emission model, shape (T, Keff)."""
        residuals = emissions[:, None, :] - self._predicted(params, x, inputs)
        return MVN(covariance_matrix=_full_cov(params.covs)).log_prob(residuals)

    def expected_log_likelihoods(self, params: ParamsSLDSEmissions, emissions: Array, qx, inputs: Array) -> Array:
        """Compute the expected log likelihoods under the continuous posterior, shape (T, Keff): the log
        likelihood at the posterior mean minus a correction for the posterior covariance."""
        plug_in = self.log_likelihoods(params, emissions, qx.smoothed_means, inputs)
        R_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        correction = 0.5 * jnp.einsum("knd,tde,kne->tk",
                                      jnp.einsum("kij,kjd->kid", R_inv, params.weights), qx.smoothed_covariances, params.weights)
        return plug_in - correction

    def expected_log_joint_x(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute the log likelihood of the emissions at ``x``, weighted by :meth:`state_weights` and summed."""
        return jnp.sum(self.state_weights(smoothed_probs) * self.log_likelihoods(params, emissions, x, inputs))

    def neg_hessian_blocks(self, params: ParamsSLDSEmissions, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to each continuous state,
        shape (T, D, D). It does not depend on ``x``, because the emissions are linear in it."""
        R_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        HRH = jnp.einsum("knd,knm,kme->kde", params.weights, R_inv, params.weights)
        return jnp.einsum("tk,kde->tde", self.state_weights(smoothed_probs), HRH)

    def invert(self, params: ParamsSLDSEmissions, emissions: Array, inputs: Array) -> Array:
        """Estimate the continuous states from the emissions by least squares, shape (T, D), with the parameters
        averaged over the emission models."""
        H, Dm, d = params.weights.mean(0), params.input_weights.mean(0), params.biases.mean(0)
        residuals = emissions - inputs @ Dm.T - d
        return residuals @ jnp.linalg.pinv(H).T

    def expected_emissions(self, params: ParamsSLDSEmissions, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the mean emission at each time step at the continuous states ``means``, weighted by
        :meth:`state_weights`, shape (T, N)."""
        return jnp.einsum("tk,tkn->tn", self.state_weights(smoothed_probs), self._predicted(params, means, inputs))

    def sample_emission(self, key: PRNGKeyT, params: ParamsSLDSEmissions, state: Array, x: Array, inpt: Array) -> Array:
        """Sample one emission given the discrete state, the continuous state and the input.

        Args:
            key: random number generator
            params: emission parameters
            state: discrete state
            x: continuous state, shape (D,)
            inpt: input, shape (M,)

        Returns:
            emission, shape (N,)
        """
        k = 0 if self.single_subspace else state
        mean = params.weights[k] @ x + params.input_weights[k] @ inpt + params.biases[k]
        cov = _full_cov(params.covs)[k]
        return mean + jnp.linalg.cholesky(symmetrize(cov)) @ jr.normal(key, (self.emission_dim,))

    # ---- sufficient statistics and M-step ---------------------------------------------------
    def collect_suff_stats(self, params: ParamsSLDSEmissions, emissions: Array, smoothed_probs: Array, qx, inputs: Array) -> EmissionStats:
        """Collect the expected sufficient statistics of a regression of the emissions on the continuous
        state, the inputs and a constant, weighted by :meth:`state_weights`."""
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
        """Return the M-step state, which is None because the M-step has a closed form."""
        return None

    def m_step(self, params: ParamsSLDSEmissions, props: ParamsSLDSEmissions, batch_stats: EmissionStats, m_step_state: Any
               ) -> Tuple[ParamsSLDSEmissions, Any]:
        """Update the parameters in closed form: penalized least squares for the weights, input weights and
        biases, then the covariances under the inverse-Wishart prior. The parameters must be all trainable or
        all frozen; nothing changes when they are frozen."""
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
