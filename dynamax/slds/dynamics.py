"""
Linear Gaussian dynamics for a switching linear dynamical system (SLDS).

Each discrete state has its own dynamics weights, bias, input weights and noise covariance. The
initial continuous state has one distribution shared by all discrete states; by default it is a
standard normal and is not learned. The M-step uses weak priors that pull the dynamics weights toward
the identity and the biases and input weights toward zero.
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
from dynamax.slds.abstractions import DynamicsStats, SLDSDynamics
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import psd_solve, random_rotation, symmetrize


class ParamsSLDSDynamics(NamedTuple):
    """Parameters of the linear Gaussian dynamics of an SLDS, one set per discrete state.

    :param initial_mean: mean of the initial continuous state, shared by all states
    :param initial_cov: covariance of the initial continuous state, shared by all states
    :param weights: dynamics weights, shape (K, D, D)
    :param biases: dynamics biases, shape (K, D)
    :param input_weights: dynamics input weights, shape (K, D, M)
    :param covs: dynamics covariances, shape (K, D, D), or (K, D) when diagonal
    """
    initial_mean: Union[Float[Array, " state_dim"], ParameterProperties]
    initial_cov: Union[Float[Array, "state_dim state_dim"], ParameterProperties]
    weights: Union[Float[Array, "num_states state_dim state_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_states state_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "num_states state_dim state_dim"], Float[Array, "num_states state_dim"], ParameterProperties]


def _full_cov(cov: Array) -> Array:
    """Expand diagonal covariances (K, D) to full ones (K, D, D); full covariances pass through unchanged."""
    return cov if cov.ndim == 3 else vmap(jnp.diag)(cov)


def _cov_inverse_and_logdet(cov: Array) -> Tuple[Array, Array]:
    """Inverse and log determinant of one (D, D) covariance."""
    chol = jnp.linalg.cholesky(symmetrize(cov))
    inv = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, jnp.eye(cov.shape[-1], dtype=cov.dtype)))
    return symmetrize(inv), 2 * jnp.log(jnp.diag(chol)).sum()


class LinearGaussianSLDSDynamics(SLDSDynamics):
    """Linear Gaussian dynamics with one set of parameters per discrete state.

    :param num_states: number of discrete states K
    :param state_dim: dimension D of the continuous state
    :param input_dim: dimension M of the inputs (may be 0)
    :param cov_type: ``"full"`` or ``"diag"`` noise covariance
    :param l2_penalty_weights: strength of the prior pulling the dynamics weights toward the identity
    :param l2_penalty_input_weights: strength of the prior pulling the input weights toward zero
    :param l2_penalty_biases: strength of the prior pulling the biases toward zero
    :param nu0: degrees of freedom of the inverse-Wishart prior on the noise covariances. Set
        ``nu0 = -(state_dim + 1)``, with ``Psi0`` and the penalties at zero, to turn the prior off
    :param Psi0: scale of the inverse-Wishart prior, as a multiple of the identity
    """
    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 input_dim: int = 0,
                 cov_type: str = "full",
                 l2_penalty_weights: float = 1e-8,
                 l2_penalty_input_weights: float = 1e-8,
                 l2_penalty_biases: float = 1e-8,
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
        self.l2_penalty_biases = l2_penalty_biases
        self.nu0 = nu0
        self.Psi0 = Psi0

    # ---- parameters -------------------------------------------------------------------------
    def _ridge(self) -> Tuple[Array, Array, Array]:
        """Prior of the dynamics regression, for each discrete state.

        Returns its precision (P, P), its precision times the prior mean (P, D), and the prior mean
        (D, P), which is the identity on the previous state and zero on the input and the constant.
        """
        D, M = self.state_dim, self.input_dim
        J0 = jnp.diag(jnp.concatenate([self.l2_penalty_weights * jnp.ones(D),
                                       self.l2_penalty_input_weights * jnp.ones(M),
                                       self.l2_penalty_biases * jnp.ones(1)]))
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
        """Initialize the dynamics parameters and their properties.

        Defaults: weights 0.95 times a random rotation (needs ``key``), zero biases and input weights, identity
        covariances, and a standard normal initial state that is not learned. Arguments passed in replace them.
        ``method`` is not used.
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
        """Compute the log prior of the dynamics parameters, up to a constant. The M-step maximizes it plus
        the expected log likelihood."""
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
        """Predicted mean of the next continuous state under each discrete state, shape (T', K, D)."""
        return (jnp.einsum("kij,tj->tki", params.weights, x_prev)
                + jnp.einsum("kim,tm->tki", params.input_weights, inputs)
                + params.biases[None])

    def ar_log_likelihoods(self, params: ParamsSLDSDynamics, x_prev: Array, x: Array, inputs: Array) -> Array:
        """Compute the log density of each continuous state given the previous one, under each discrete state.

        Args:
            params: dynamics parameters
            x_prev: previous continuous states, shape (T', D)
            x: continuous states, shape (T', D)
            inputs: inputs at the same time steps as ``x``, shape (T', M)

        Returns:
            log densities, shape (T', K)
        """
        residuals = x[:, None, :] - self._predicted_means(params, x_prev, inputs)   # (T', K, D)
        return MVN(covariance_matrix=_full_cov(params.covs)).log_prob(residuals)

    def log_likelihoods(self, params: ParamsSLDSDynamics, x: Array, inputs: Array) -> Array:
        """Compute the log density of the continuous states under each discrete state's dynamics, shape (T, K)."""
        row0 = MVN(params.initial_mean, params.initial_cov).log_prob(x[0])
        rest = self.ar_log_likelihoods(params, x[:-1], x[1:], inputs[1:])
        return jnp.concatenate([jnp.full((1, self.num_states), row0), rest], axis=0)

    def expected_log_likelihoods(self, params: ParamsSLDSDynamics, qx, inputs: Array) -> Array:
        """Compute the log densities of :meth:`log_likelihoods` in exact expectation under the continuous
        posterior, shape (T, K)."""
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
            """Expected log densities for one discrete state, shape (T-1,)."""
            M = (jnp.einsum("ti,tj->tij", res_k, res_k) + covs[1:]
                 - jnp.einsum("ij,tjk->tik", F, centered_cross)
                 - jnp.einsum("tji,kj->tik", centered_cross, F)
                 + jnp.einsum("ij,tjk,lk->til", F, covs[:-1], F))
            return -0.5 * (D * jnp.log(2 * jnp.pi) + Q_logdet_k + jnp.einsum("ij,tji->t", Q_inv_k, M))

        rest = vmap(_one_state, in_axes=(0, 0, 0, 1), out_axes=1)(params.weights, Q_inv, Q_logdet, residuals)
        return jnp.concatenate([jnp.full((1, self.num_states), row0), rest], axis=0)

    def expected_log_joint_x(self, params: ParamsSLDSDynamics, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute the log density of the continuous states, weighted by the discrete posterior and summed."""
        return jnp.sum(smoothed_probs * self.log_likelihoods(params, x, inputs))

    def neg_hessian_blocks(self, params: ParamsSLDSDynamics, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to the continuous states.

        It does not depend on the continuous states, and ``inputs`` are not used.
        """
        num_timesteps = smoothed_probs.shape[0]
        Q_inv, _ = vmap(_cov_inverse_and_logdet)(_full_cov(params.covs))
        FQF = jnp.einsum("kji,kjl,klm->kim", params.weights, Q_inv, params.weights)   # F^T Q^{-1} F
        QF = jnp.einsum("kij,kjl->kil", Q_inv, params.weights)                         # Q^{-1} F
        w_next = smoothed_probs[1:]                                                    # smoothed_probs[t+1]
        S0_inv, _ = _cov_inverse_and_logdet(params.initial_cov)
        J_diag = jnp.zeros((num_timesteps, self.state_dim, self.state_dim)).at[0].set(S0_inv)
        J_diag = J_diag.at[:-1].add(jnp.einsum("tk,kij->tij", w_next, FQF))
        J_diag = J_diag.at[1:].add(jnp.einsum("tk,kij->tij", w_next, Q_inv))
        J_lower = -jnp.einsum("tk,kij->tij", w_next, QF)
        return J_diag, J_lower

    def sample_transition(self, key: PRNGKeyT, params: ParamsSLDSDynamics, state: Array, x_prev: Array, inpt: Array) -> Array:
        """Sample the next continuous state given the discrete state, the previous continuous state and the input.

        Args:
            key: random number generator
            params: dynamics parameters
            state: discrete state
            x_prev: previous continuous state, shape (D,)
            inpt: input, shape (M,)

        Returns:
            continuous state, shape (D,)
        """
        mean = params.weights[state] @ x_prev + params.input_weights[state] @ inpt + params.biases[state]
        cov = _full_cov(params.covs)[state]
        return mean + jnp.linalg.cholesky(symmetrize(cov)) @ jr.normal(key, (self.state_dim,))

    # ---- sufficient statistics --------------------------------------------------------------
    def _regression_stats(self, weights, x_prev, x, inputs, cov_prev=None, centered_cross=None, cov=None):
        """Weighted second moments of the regressors and the next continuous state, per discrete state.

        The regressors are the previous continuous state, the input and a constant. When the
        posterior covariances are given, they are added so that the moments are expectations under
        the continuous posterior. The moments of the initial state are left at zero.
        """
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
        """Collect the expected sufficient statistics of the dynamics under the continuous posterior.

        Each pair of consecutive time steps is weighted by the posterior probability of the discrete state at the
        later step.
        """
        means, covs, crosses = qx.smoothed_means, qx.smoothed_covariances, qx.smoothed_cross_covariances
        centered_cross = crosses - jnp.einsum("ti,tj->tij", means[:-1], means[1:])
        stats = self._regression_stats(smoothed_probs[1:], means[:-1], means[1:], inputs[1:], covs[:-1], centered_cross, covs[1:])
        return stats._replace(Ex0=means[0], Ex0x0T=covs[0] + jnp.outer(means[0], means[0]))

    # ---- M-step -----------------------------------------------------------------------------
    def initialize_m_step_state(self, params: ParamsSLDSDynamics, props: ParamsSLDSDynamics) -> Any:
        """Return the M-step state, which is None because the M-step has a closed form."""
        return None

    def m_step(self,
               params: ParamsSLDSDynamics,
               props: ParamsSLDSDynamics,
               batch_stats: DynamicsStats,
               m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParamsSLDSDynamics, Any]:
        """Update the parameters in closed form to maximize the expected log likelihood plus the log prior.

        Each state's weights, biases and input weights come from regularized least squares, its covariance from
        the remaining error. A trainable initial state takes the averaged posterior moments. The weights, biases,
        input weights and covariances must be all trainable or all frozen, and likewise the initial mean and
        covariance. With ``key``, an unused state is replaced by a slightly perturbed copy of a used one.
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
        """Replace each discrete state with less than one expected observation.

        The replacement copies a randomly chosen state with more than one expected observation and
        adds noise with standard deviation 0.01 to its weights, input weights and biases; the noise
        covariance is copied as is. Nothing changes when no state has more than one.
        """
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
