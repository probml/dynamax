"""
Interfaces of the components of a switching linear dynamical system (SLDS), and the base class
:class:`SLDS` that combines and fits them.
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
    """Expected sufficient statistics of the dynamics, weighted by the posterior probability of each discrete state.

    The dynamics are a regression of each continuous state on the previous continuous state, the input and a
    constant (the P regressors). Every time step after the first adds to the sums; the first time step gives
    the moments of the initial continuous state.

    :param Ex0: expected initial continuous state, shape (D,)
    :param Ex0x0T: expected outer product of the initial continuous state, shape (D, D)
    :param sum_w: total posterior probability of each discrete state, shape (K,)
    :param sum_phiphiT: weighted sum of the outer products of the regressors, shape (K, P, P)
    :param sum_phixT: weighted sum of the outer products of the regressors and the continuous state, shape (K, P, D)
    :param sum_xxT: weighted sum of the outer products of the continuous state, shape (K, D, D)
    """
    Ex0: Float[Array, " state_dim"]
    Ex0x0T: Float[Array, "state_dim state_dim"]
    sum_w: Float[Array, " num_states"]
    sum_phiphiT: Float[Array, "num_states num_regressors num_regressors"]
    sum_phixT: Float[Array, "num_states num_regressors state_dim"]
    sum_xxT: Float[Array, "num_states state_dim state_dim"]


class EmissionStats(NamedTuple):
    """Expected sufficient statistics of the emissions, weighted by the posterior probability of each discrete state.

    The emissions are a regression on the continuous state, the input and a constant (the P regressors). With
    a single subspace all states share one emission model, every weight is one and the leading axis has size 1.

    :param sum_w: total weight of each emission model, shape (K,) or (1,)
    :param sum_psipsiT: weighted sum of the outer products of the regressors, shape (K, P, P)
    :param sum_psiyT: weighted sum of the outer products of the regressors and the emission, shape (K, P, N)
    :param sum_yyT: weighted sum of the outer products of the emission, shape (K, N, N)
    """
    sum_w: Float[Array, " num_subspaces"]
    sum_psipsiT: Float[Array, "num_subspaces num_regressors num_regressors"]
    sum_psiyT: Float[Array, "num_subspaces num_regressors emission_dim"]
    sum_yyT: Float[Array, "num_subspaces emission_dim emission_dim"]


class RecurrentTransStats(NamedTuple):
    """Statistics of one sequence for the M-step of transitions that depend on the continuous state.

    :param trans_probs: posterior probability of each pair of consecutive discrete states, shape (T-1, K, K)
    :param x_samples: samples of the continuous states from the posterior, shape (S, T, D)
    :param inputs: inputs of the sequence, shape (T, M)
    """
    trans_probs: Float[Array, "num_timesteps_minus_1 num_states num_states"]
    x_samples: Float[Array, "num_samples num_timesteps state_dim"]
    inputs: Float[Array, "num_timesteps input_dim"]


class SuffStatsSLDS(NamedTuple):
    """Expected sufficient statistics of one sequence.

    :param initial: posterior probability of each discrete state at the first time step, shape (K,)
    :param transitions: statistics of the transition component
    :param dynamics: statistics of the dynamics component
    :param emissions: statistics of the emission component
    """
    initial: Float[Array, " num_states"]
    transitions: Any
    dynamics: DynamicsStats
    emissions: EmissionStats


# ---- component interfaces -------------------------------------------------------------------
class SLDSTransitions(ABC):
    """Interface of the transition distribution of the discrete state.

    The next discrete state may depend on the current discrete state, the current continuous state and the
    next input. Subclasses whose transitions depend on the continuous state set the class attribute
    ``depends_on_x`` to True; the model then uses samples of the continuous states to update the discrete
    posterior, and updates the continuous posterior iteratively instead of in one step.
    """
    depends_on_x: bool = False

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute the transition matrices of the discrete state.

        The shape is (K, K) when the transitions are stationary, (T-1, K, K) otherwise, with one matrix
        per pair of consecutive time steps.
        """

    @abstractmethod
    def log_transition_matrices(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute the log transition matrices, with normalized rows, in the same shape as
        :meth:`transition_matrices`."""

    @abstractmethod
    def transition_row(self, params: ParameterSet, state: Array, x: Array, next_input: Array) -> Array:
        """Compute the distribution of the next discrete state, shape (K,), used for sampling.

        It is conditioned on the current discrete state ``state``, the current continuous state ``x`` and the
        next input ``next_input``.
        """

    @abstractmethod
    def expected_log_trans_prob(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """Compute the expected log probability of the discrete transitions at the continuous states ``x``.

        Each log transition probability is weighted by the posterior probability ``trans_probs`` of that pair
        of consecutive discrete states, and the result is summed over time steps and pairs.
        """

    def neg_hessian_x(self, params: ParameterSet, x: Array, inputs: Array, trans_probs: Array) -> Array:
        """Compute the negative Hessian of :meth:`expected_log_trans_prob` with respect to each continuous state.

        The shape is (T-1, D, D). It is added to the first T-1 diagonal blocks of the precision of the
        continuous posterior.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, qz: DiscretePosteriorSLDS, x_samples: Array, inputs: Array) -> Any:
        """Collect the statistics of one sequence that the M-step needs."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps, such as an optimizer state, or None."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: Any, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """Update the parameters from statistics batched over sequences along a leading axis."""


class SLDSDynamics(ABC):
    """Interface of the dynamics of the continuous state, and of its initial distribution."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, x: Array, inputs: Array) -> Array:
        """Compute the log density of the continuous states under each discrete state's dynamics, shape (T, K).

        Row 0 holds the initial density, which is the same for every discrete state, and the other rows the
        transition densities.
        """

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """Compute the log densities of :meth:`log_likelihoods` in expectation under the continuous posterior,
        shape (T, K)."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute the log density of the continuous states ``x``, weighted by the posterior probability of each
        discrete state and summed over time steps and states."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, smoothed_probs: Array, inputs: Array) -> Tuple[Array, Array]:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to the continuous states.

        Returns:
            tuple of the diagonal blocks, shape (T, D, D), and the blocks just below the diagonal, shape (T-1, D, D)
        """

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> DynamicsStats:
        """Compute the expected sufficient statistics of the dynamics under the continuous posterior."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: DynamicsStats, m_step_state: Any,
               key: Optional[PRNGKeyT] = None) -> Tuple[ParameterSet, Any]:
        """Update the parameters from statistics batched over sequences along a leading axis."""


class SLDSEmissions(ABC):
    """Interface of the emission distribution, given the continuous state, the discrete state and the input.

    With ``single_subspace`` True, all discrete states share one emission model, and the methods return one
    column of per-state values instead of K.
    """
    single_subspace: bool = True

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        """Shape of one emission."""

    @abstractmethod
    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior", **kwargs) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the parameters and their corresponding properties."""

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the parameters."""

    @abstractmethod
    def log_likelihoods(self, params: ParameterSet, emissions: Array, x: Array, inputs: Array) -> Array:
        """Compute the log likelihood of each emission under each emission model, shape (T, K), or (T, 1) with a
        single subspace."""

    @abstractmethod
    def expected_log_likelihoods(self, params: ParameterSet, emissions: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> Array:
        """Compute the log likelihoods of :meth:`log_likelihoods` in expectation under the continuous posterior,
        shape (T, K), or (T, 1) with a single subspace."""

    @abstractmethod
    def expected_log_joint_x(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Scalar:
        """Compute the log likelihood of the emissions at the continuous states ``x``, weighted by
        :meth:`state_weights` and summed over time steps and emission models."""

    @abstractmethod
    def neg_hessian_blocks(self, params: ParameterSet, emissions: Array, x: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the negative Hessian of :meth:`expected_log_joint_x` with respect to each continuous state,
        shape (T, D, D)."""

    @abstractmethod
    def invert(self, params: ParameterSet, emissions: Array, inputs: Array) -> Array:
        """Estimate the continuous states from the emissions, to center the starting continuous posterior."""

    @abstractmethod
    def state_weights(self, smoothed_probs: Array) -> Array:
        """Compute the weight of each emission model at each time step.

        The weights are ``smoothed_probs``, shape (T, K), or a column of ones, shape (T, 1), with a single subspace.
        """

    @abstractmethod
    def expected_emissions(self, params: ParameterSet, means: Array, smoothed_probs: Array, inputs: Array) -> Array:
        """Compute the expected emissions at the continuous states ``means``, averaged over the emission models
        with :meth:`state_weights`, shape (T, N). Used by :meth:`SLDS.smooth`."""

    @abstractmethod
    def collect_suff_stats(self, params: ParameterSet, emissions: Array, smoothed_probs: Array, qx: ContinuousPosteriorSLDS, inputs: Array) -> EmissionStats:
        """Compute the expected sufficient statistics of the emissions under the continuous posterior."""

    @abstractmethod
    def initialize_m_step_state(self, params: ParameterSet, props: PropertySet) -> Any:
        """Initialize the state carried between M-steps."""

    @abstractmethod
    def m_step(self, params: ParameterSet, props: PropertySet, batch_stats: EmissionStats, m_step_state: Any) -> Tuple[ParameterSet, Any]:
        """Update the parameters from statistics batched over sequences along a leading axis."""


class SLDSInitialState(StandardHMMInitialState):
    """Initial distribution of the discrete state, with a Dirichlet prior.

    The M-step clips the posterior mode at zero and adds a small floor, proportional to the total count, so
    that states with no expected counts keep a nonzero probability.

    :param num_states: number of discrete states
    :param initial_probs_concentration: concentration of the Dirichlet prior
    """

    def __init__(self, num_states: int, initial_probs_concentration: Union[Scalar, Array] = 1.0):
        super().__init__(num_states, initial_probs_concentration)

    def log_prior(self, params: ParamsStandardHMMInitialState) -> Scalar:
        """Compute the log prior probability of the parameters, or zero for a single state, where the
        Dirichlet is undefined."""
        if self.num_states == 1:
            return 0.0
        return super().log_prior(params)

    def m_step(self, params, props, batch_stats, m_step_state):
        """Set the initial probabilities to the posterior mode, clipped at zero, floored and normalized."""
        if props.probs.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.initial_probs_concentration - 1.0), 0.0)
            counts = counts + 1e-8 * jnp.maximum(counts.sum(), 1.0)
            params = params._replace(probs=counts / counts.sum())
        return params, m_step_state


# ---- parameters ----------------------------------------------------------------------------
class ParamsSLDS(NamedTuple):
    """Parameters of an SLDS.

    :param initial: initial distribution parameters
    :param transitions: transition distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters
    """
    initial: Any
    transitions: Any
    dynamics: Any
    emissions: Any
