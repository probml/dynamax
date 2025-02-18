"""
This module contains the implementation of the initial distribution of a hidden Markov model.
"""
from typing import Any, cast, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMMInitialState
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar


class ParamsStandardHMMInitialState(NamedTuple):
    """Named tuple for the parameters of the standard HMM initial distribution."""
    probs: Union[Float[Array, " state_dim"], ParameterProperties]


class StandardHMMInitialState(HMMInitialState):
    """Abstract class for HMM initial distributions.
    """
    def __init__(self,
                 num_states: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1):
        """
        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
        """
        self.num_states = num_states
        self.initial_probs_concentration = initial_probs_concentration * jnp.ones(num_states)

    def distribution(self, params: ParamsStandardHMMInitialState, inputs=None) -> tfd.Distribution:
        """Return the distribution object of the initial distribution."""
        return tfd.Categorical(probs=params.probs)

    def initialize(
            self,
            key: Optional[Array]=None,
            method="prior",
            initial_probs: Optional[Float[Array, " num_states"]]=None
    ) -> Tuple[ParamsStandardHMMInitialState, ParamsStandardHMMInitialState]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            initial_probs (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Initialize the initial probabilities
        if initial_probs is None:
            if key is None:
                raise ValueError("key must be provided if initial_probs is not provided.")
            else:
                this_key, key = jr.split(key)
                initial_probs = tfd.Dirichlet(self.initial_probs_concentration).sample(seed=this_key)

        # Package the results into dictionaries
        params = ParamsStandardHMMInitialState(probs=initial_probs)
        props = ParamsStandardHMMInitialState(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStandardHMMInitialState) -> Scalar:
        """Compute the log prior of the parameters."""
        return tfd.Dirichlet(self.initial_probs_concentration).log_prob(params.probs)

    def _compute_initial_probs(
            self, params: ParamsStandardHMMInitialState, inputs=None
            ) -> Float[Array, " num_states"]:
        """Compute the initial probabilities."""
        return params.probs

    def collect_suff_stats(self, params, posterior: HMMPosterior, inputs=None) -> Float[Array, " num_states"]:
        """Collect the sufficient statistics for the initial distribution."""
        return posterior.smoothed_probs[0]

    def initialize_m_step_state(self, params, props) -> None:
        """Initialize the state for the M-step."""
        return None

    def m_step(
            self,
            params: ParamsStandardHMMInitialState,
            props: ParamsStandardHMMInitialState,
            batch_stats: Float[Array, "batch num_states"],
            m_step_state: Any
    ) -> Tuple[ParamsStandardHMMInitialState, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.probs.trainable:
            if self.num_states == 1:
                probs = jnp.array([1.0])
            else:
                expected_initial_counts = batch_stats.sum(axis=0)
                probs = tfd.Dirichlet(self.initial_probs_concentration + expected_initial_counts).mode()
            params = params._replace(probs=probs)
        return params, m_step_state

