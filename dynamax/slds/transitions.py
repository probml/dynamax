r"""
Discrete transition components of a switching linear dynamical system.

:class:`StationarySLDSTransitions` is a fixed transition matrix (the plain SLDS).
"""
from typing import Any, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.scipy.special import xlogy
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties
from dynamax.slds.abstractions import DiscretePosteriorSLDS, SLDSTransitions
from dynamax.types import PRNGKeyT, Scalar

ParamsStationarySLDSTransitions = ParamsStandardHMMTransitions


class StationarySLDSTransitions(SLDSTransitions):
    r"""A fixed transition matrix :math:`P` with a Dirichlet prior on its rows,
    :math:`P_k \sim \mathrm{Dir}(\beta 1_K + \kappa e_k)` (``ssm``: ``StationaryTransitions`` at
    :math:`\beta = 1, \kappa = 0`, ``StickyTransitions`` with :math:`\kappa > 0`).

    :param num_states: :math:`K`
    :param concentration: :math:`\beta`; 1.0 (the default) is the maximum likelihood update
    :param stickiness: :math:`\kappa`
    """
    depends_on_x = False

    def __init__(self, num_states: int, concentration: Union[Scalar, Array] = 1.0, stickiness: Union[Scalar, Array] = 0.0):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)

    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior",
                   transition_matrix: Optional[Array] = None) -> Tuple[ParamsStationarySLDSTransitions, ParamsStationarySLDSTransitions]:
        """A Dirichlet draw per row (``method="prior"``), ``ssm``'s sticky random start
        :math:`0.95 I + 0.05 U` row-normalized (``method="sticky"``, used by the ARHMM initialization),
        or a user-supplied matrix; and the properties."""
        if transition_matrix is None:
            if key is None:
                raise ValueError("key must be provided if transition_matrix is not provided.")
            if method == "prior":
                transition_matrix = jr.dirichlet(key, self.concentration)
            elif method == "sticky":
                K = self.num_states
                transition_matrix = 0.95 * jnp.eye(K) + 0.05 * jr.uniform(key, (K, K))
                transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"Unknown initialization method {method!r}")
        params = ParamsStationarySLDSTransitions(transition_matrix=transition_matrix)
        props = ParamsStationarySLDSTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStationarySLDSTransitions) -> Scalar:
        """Dirichlet log prior of the rows; zero for a single state."""
        if self.num_states == 1:
            return 0.0
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def transition_matrices(self, params, x, inputs) -> Array:
        """The (K, K) matrix."""
        return params.transition_matrix

    def log_transition_matrices(self, params, x, inputs) -> Array:
        """log of the (K, K) matrix."""
        return jnp.log(params.transition_matrix)

    def transition_row(self, params, state, x, next_input) -> Array:
        """Row ``state`` of the matrix."""
        return params.transition_matrix[state]

    def expected_log_trans_prob(self, params, x, inputs, trans_probs) -> Scalar:
        """sum_{ij} Ezzp1[i, j] log P[i, j], with the pairwise expectations summed over time."""
        return jnp.sum(xlogy(trans_probs, params.transition_matrix))

    def neg_hessian_x(self, params, x, inputs, trans_probs) -> Array:
        """Zero: the term does not depend on x."""
        num_timesteps, state_dim = x.shape
        return jnp.zeros((num_timesteps - 1, state_dim, state_dim))

    def collect_suff_stats(self, params, qz: DiscretePosteriorSLDS, x_samples, inputs) -> Array:
        """The (K, K) expected transition counts."""
        return qz.trans_probs

    def initialize_m_step_state(self, params, props) -> Any:
        return None

    def m_step(self, params, props, batch_stats, m_step_state) -> Tuple[ParamsStationarySLDSTransitions, Any]:
        r"""Floored normalized counts, :math:`P_{ij} \propto \max(\sum_b \mathbb{E}[n_{ij}] + \beta_{ij} - 1, 0) + 10^{-32} n_i`
        with :math:`n_i` the row total (``ssm``'s ``StationaryTransitions.m_step`` with its ``1e-32`` floor made relative,
        so it is not flushed to zero in float32; the Dirichlet mode where it is defined)."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.ones((1, 1))
            else:
                counts = jnp.maximum(batch_stats.sum(axis=0) + (self.concentration - 1.0), 0.0)
                counts = counts + 1e-32 * jnp.maximum(counts.sum(axis=-1, keepdims=True), 1.0)
                transition_matrix = counts / counts.sum(axis=-1, keepdims=True)
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state

    def as_hmm_transitions(self, state_dim: int):
        """The ``HMMTransitions`` adapter used by the ARHMM initialization."""
        from dynamax.slds.initialization import _ARHMMTransitions
        return _ARHMMTransitions(self, state_dim)

