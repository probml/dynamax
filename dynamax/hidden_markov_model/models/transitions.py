"""Module for HMM transition models."""
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.parameters import ParameterProperties
from dynamax.types import IntScalar, Scalar

from jaxtyping import Float, Array
from typing import Any, cast, NamedTuple, Optional, Tuple, Union


class ParamsStandardHMMTransitions(NamedTuple):
    """Named tuple for the parameters of the StandardHMMTransitions model."""
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]


class StandardHMMTransitions(HMMTransitions):
    r"""Standard model for HMM transitions.

    We place a Dirichlet prior over the rows of the transition matrix $A$,

    $$A_k \sim \mathrm{Dir}(\beta 1_K + \kappa e_k)$$

    where

    * $1_K$ denotes a length-$K$ vector of ones,
    * $e_k$ denotes the one-hot vector with a 1 in the $k$-th position,
    * $\beta \in \mathbb{R}_+$ is the concentration, and
    * $\kappa \in \mathbb{R}_+$ is the `stickiness`.

    """
    def __init__(
            self,
            num_states: int,
            concentration: Union[Scalar, Float[Array, "num_states num_states"]]=1.1,
            stickiness: Union[Scalar, Float[Array, " num_states"]]=0.0
    ):
        """
        Args:
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        self.num_states = num_states
        self.concentration = \
            concentration * jnp.ones((num_states, num_states)) + \
                stickiness * jnp.eye(num_states)

    def distribution(self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None):
        """Return the distribution over the next state given the current state."""
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
            self,
            key: Optional[Array]=None,
            method="prior",
            transition_matrix: Optional[Float[Array, "num_states num_states"]]=None
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            transition_matrix (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if transition_matrix is None:
            if key is None:
                raise ValueError("key must be provided if transition_matrix is not provided.")
            else:
                transition_matrix_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                transition_matrix = cast(Float[Array, "num_states num_states"], transition_matrix_sample)

        # Package the results into dictionaries
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        """Compute the log prior probability of the parameters."""
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def _compute_transition_matrices(
            self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        """Compute the transition matrices."""
        return params.transition_matrix

    def collect_suff_stats(
            self,
            params,
            posterior: HMMPosterior,
            inputs=None
    ) -> Union[Float[Array, "num_states num_states"],
               Float[Array, "num_timesteps_minus_1 num_states num_states"]]:
        """Collect the sufficient statistics for the model."""
        return posterior.trans_probs

    def initialize_m_step_state(self, params, props):
        """Initialize the state for the M-step."""
        return None

    def m_step(
            self,
            params: ParamsStandardHMMTransitions,
            props: ParamsStandardHMMTransitions,
            batch_stats: Float[Array, "batch num_states num_states"],
            m_step_state: Any
        ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats.sum(axis=0)
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state
