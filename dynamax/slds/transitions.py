"""
Transitions between the discrete states of a switching linear dynamical system (SLDS).

:class:`StationarySLDSTransitions` uses a fixed transition matrix, as in the standard SLDS.
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
    """A fixed transition matrix with a Dirichlet prior on each row.

    :param num_states: number of discrete states
    :param concentration: Dirichlet concentration of each row; the default of 1.0 gives the maximum likelihood update
    :param stickiness: extra concentration on the diagonal, which favors staying in the same state
    """
    depends_on_x = False

    def __init__(self, num_states: int, concentration: Union[Scalar, Array] = 1.0, stickiness: Union[Scalar, Array] = 0.0):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)

    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior",
                   transition_matrix: Optional[Array] = None) -> Tuple[ParamsStationarySLDSTransitions, ParamsStationarySLDSTransitions]:
        """A Dirichlet draw per row (``method="prior"``) or a user-supplied matrix; and the properties."""
        if transition_matrix is None:
            if key is None:
                raise ValueError("key must be provided if transition_matrix is not provided.")
            if method == "prior":
                transition_matrix = jr.dirichlet(key, self.concentration)
            else:
                raise ValueError(f"Unknown initialization method {method!r}")
        params = ParamsStationarySLDSTransitions(transition_matrix=transition_matrix)
        props = ParamsStationarySLDSTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStationarySLDSTransitions) -> Scalar:
        """Compute the log prior of the transition matrix; it is zero when there is a single state."""
        if self.num_states == 1:
            return 0.0
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def transition_matrices(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array) -> Array:
        """Return the transition matrix, shape (K, K)."""
        return params.transition_matrix

    def log_transition_matrices(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array) -> Array:
        """Return the log of the transition matrix, shape (K, K)."""
        return jnp.log(params.transition_matrix)

    def transition_row(self, params: ParamsStationarySLDSTransitions, state: Array, x: Array, next_input: Array) -> Array:
        """Return the probabilities of the next discrete state given the current one, shape (K,)."""
        return params.transition_matrix[state]

    def expected_log_trans_prob(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """Compute the expected log probability of the transitions under the discrete posterior.

        ``trans_probs`` holds the posterior probability of each pair of consecutive states, summed over time.
        """
        return jnp.sum(xlogy(trans_probs, params.transition_matrix))

    def collect_suff_stats(self, params: ParamsStationarySLDSTransitions, qz: DiscretePosteriorSLDS, x_samples: Array,
                           inputs: Array) -> Array:
        """Return the expected number of transitions between each pair of states, shape (K, K)."""
        return qz.trans_probs

    def initialize_m_step_state(self, params: ParamsStationarySLDSTransitions, props: ParamsStationarySLDSTransitions) -> Any:
        return None

    def m_step(self,
               params: ParamsStationarySLDSTransitions,
               props: ParamsStationarySLDSTransitions,
               batch_stats: Array,
               m_step_state: Any) -> Tuple[ParamsStationarySLDSTransitions, Any]:
        """Update the transition matrix to the mode of its posterior given the expected counts.

        The counts plus the prior are clipped at zero, and a tiny floor relative to each row's total is added so
        that no transition probability is exactly zero.
        """
        if props.transition_matrix.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.concentration - 1.0), 0.0)
            counts = counts + 1e-32 * jnp.maximum(counts.sum(axis=-1, keepdims=True), 1.0)
            transition_matrix = counts / counts.sum(axis=-1, keepdims=True)
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state
