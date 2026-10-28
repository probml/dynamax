r"""
Component interfaces and the driver of the switching linear dynamical system.

An SLDS has a discrete chain :math:`z_{1:T}` and a continuous chain :math:`x_{1:T}`:

.. math::
    z_0 \sim \mathrm{Cat}(\pi_0), \quad z_t \mid z_{t-1} \sim \mathrm{Cat}(P_t[z_{t-1}, :]), \quad
    x_0 \sim \mathcal{N}(m_0, \Sigma_0), \quad x_t \mid x_{t-1}, z_t \sim \mathcal{N}(F_{z_t} x_{t-1} + b_{z_t} + B_{z_t} u_t, Q_{z_t}), \quad
    y_t \mid x_t, z_t \sim \mathcal{N}(H_{z_t} x_t + d_{z_t} + D_{z_t} u_t, R_{z_t}), \qquad t = 0, \ldots, T-1.

Exact inference is intractable, so :class:`SLDS` fits the structured mean-field posterior
:math:`q(z) q(x)` of Laplace variational EM (Zoltowski, Pillow and Linderman, 2020), ported
from ``lindermanlab/ssm``: :math:`q(z)` is a Markov chain updated with the HMM smoother and
:math:`q(x)` a Gaussian with block-tridiagonal precision updated with a Newton step at its mode.
"""
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState, StandardHMMInitialState
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.slds.inference import ContinuousPosteriorSLDS, DiscretePosteriorSLDS
from dynamax.types import PRNGKeyT, Scalar


# ---- sufficient statistics ------------------------------------------------------------------
class DynamicsStats(NamedTuple):
    r"""Expected statistics of the dynamics regression :math:`x_t \approx \Theta_k \phi_t`,
    :math:`\phi_t = [x_{t-1}; u_t; 1]`, weighted by :math:`\mathbb{E}[z_t = k]` over :math:`t \geq 1`,
    plus the moments of :math:`x_0`."""
    Ex0: Float[Array, " state_dim"]
    Ex0x0T: Float[Array, "state_dim state_dim"]
    sum_w: Float[Array, " num_states"]
    sum_phiphiT: Float[Array, "num_states num_regressors num_regressors"]
    sum_phixT: Float[Array, "num_states num_regressors state_dim"]
    sum_xxT: Float[Array, "num_states state_dim state_dim"]


class EmissionStats(NamedTuple):
    r"""Expected statistics of the emission regression :math:`y_t \approx \Theta_k \psi_t`,
    :math:`\psi_t = [x_t; u_t; 1]`, weighted by :math:`\mathbb{E}[z_t = k]` (or 1 with a single subspace)."""
    sum_w: Float[Array, " num_subspaces"]
    sum_psipsiT: Float[Array, "num_subspaces num_regressors num_regressors"]
    sum_psiyT: Float[Array, "num_subspaces num_regressors emission_dim"]
    sum_yyT: Float[Array, "num_subspaces emission_dim emission_dim"]


class RecurrentTransStats(NamedTuple):
    """Statistics of recurrent transitions: pairwise expectations plus the samples of x and the
    inputs the M-step objective needs."""
    trans_probs: Float[Array, "num_timesteps_minus_1 num_states num_states"]
    x_samples: Float[Array, "num_samples num_timesteps state_dim"]
    inputs: Float[Array, "num_timesteps input_dim"]


class SuffStatsSLDS(NamedTuple):
    """Expected sufficient statistics of one sequence, as returned by :meth:`SLDS.e_step`."""
    initial: Float[Array, " num_states"]
    transitions: Any
    dynamics: DynamicsStats
    emissions: EmissionStats


# ---- component interfaces -------------------------------------------------------------------
class SLDSTransitions(ABC):
    r"""Discrete transitions :math:`p(z_{t+1} \mid z_t, x_t, u_{t+1})`.

    ``depends_on_x`` is a static class attribute: recurrent transitions need Monte Carlo
    samples of :math:`x` and a Newton (not closed-form) update of :math:`q(x)`.
    """
    depends_on_x: bool = False

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """(K, K) if stationary, else (T-1, K, K) with entry t linking z_t and z_{t+1}."""

    @abstractmethod
    def log_transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Row-normalized log transition matrices, same shape as :meth:`transition_matrices`."""

    @abstractmethod
    def transition_row(self, params: ParameterSet, state: Array, x: Array, next_input: Array) -> Array:
        """p(z_{t+1} = . | z_t = state, x_t = x, u_{t+1} = next_input), shape (K,); for sampling."""

    @abstractmethod
    def expected_log_trans_prob(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """sum over t, i, j of E[z_t = i, z_{t+1} = j] log P_t(x)[i, j]."""

    @abstractmethod
    def neg_hessian_x(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Array:
        """Negative Hessian in x_t of the expected log transition term, (T-1, D, D), added to J_diag[:-1]."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, qz: DiscretePosteriorSLDS, x_samples: Array, inputs: Array) -> Any:
        """Per-sequence statistics for the M-step."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps (an optimizer state, or None)."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: Any, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics (leading batch axis)."""


class SLDSDynamics(ABC):
    r"""Continuous dynamics :math:`p(x_t \mid x_{t-1}, z_t, u_t)` and the initial distribution of :math:`x_0`."""
    is_quadratic_in_x: bool = True

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """(T, K): row 0 is the initial density for every state, rows t >= 1 the transition densities."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """(T, K): the same, in expectation under q(x)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """sum_t sum_k Ez[t, k] log p(x_t | x_{t-1}, z_t = k) at the point x."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        """(J_diag (T, D, D), J_lower (T-1, D, D)) of the negative Hessian of :meth:`expected_log_joint_x`."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> DynamicsStats:
        """Expected statistics under q(x)."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: DynamicsStats, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics."""


class SLDSEmissions(ABC):
    r"""Emissions :math:`p(y_t \mid x_t, z_t, u_t)`."""
    is_quadratic_in_x: bool = True
    single_subspace: bool = True

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Parameters and properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Log prior of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, emissions: Array, x: Array, inputs: Array, mask=None) -> Array:
        """(T, Keff) log p(y_t | x_t, z_t = k, u_t)."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, emissions: Array, qx: ContinuousPosteriorSLDS, inputs: Array, mask=None) -> Array:
        """(T, Keff): the same, in expectation under q(x)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Scalar:
        """sum_t sum_k w[t, k] log p(y_t | x_t, z_t = k) at the point x."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array, mask=None) -> Array:
        """(T, D, D) negative Hessian of :meth:`expected_log_joint_x` in x_t."""

    @abstractmethod
    def invert(self, params: ParameterSet, emissions: Array, inputs: Array) -> Array:
        """A point estimate of x_{0:T-1} from the emissions, for the initial q(x)."""

    @abstractmethod
    def state_weights(self, smoothed_probs: Array) -> Array:
        """(T, Keff) weights of the per-subspace terms: ``smoothed_probs``, or a column of ones with a single subspace."""

    @abstractmethod
    def expected_emissions(self, params: ParameterSet, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """(T, N): sum_k w[t, k] E[y_t | x_t = means[t], z_t = k, u_t]; used by :meth:`SLDS.smooth`."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, emissions: Array, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array, mask=None) -> EmissionStats:
        """Expected statistics under q(x)."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """State carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: EmissionStats, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """M-step on batched statistics."""


class SLDSInitialState(StandardHMMInitialState):
    r"""Initial discrete distribution with ``ssm``'s M-step, :math:`\pi_0 \propto \max(\sum_b \mathbb{E}[z_{b,0}] + \beta - 1, 0) + 10^{-8}`
    (the floor scaled by the total count). ``StandardHMMInitialState``'s Dirichlet-mode update is NaN
    whenever a count is zero at concentration 1."""

    def __init__(self, num_states: int, initial_probs_concentration: Union[Scalar, Array] = 1.0):
        super().__init__(num_states, initial_probs_concentration)

    def log_prior(self, params: ParamsStandardHMMInitialState) -> Scalar:
        """Dirichlet log prior; zero for a single state, where the Dirichlet is undefined."""
        if self.num_states == 1:
            return 0.0
        return super().log_prior(params)

    def m_step(self, params, props, batch_stats, m_step_state):
        """Floored, normalized expected counts."""
        if props.probs.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.initial_probs_concentration - 1.0), 0.0)
            counts = counts + 1e-8 * jnp.maximum(counts.sum(), 1.0)
            params = params._replace(probs=counts / counts.sum())
        return params, m_step_state


# ---- parameters ----------------------------------------------------------------------------
class ParamsSLDS(NamedTuple):
    """Parameters of an SLDS: one entry per component."""
    initial: Any
    transitions: Any
    dynamics: Any
    emissions: Any
