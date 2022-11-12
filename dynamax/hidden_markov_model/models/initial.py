from dynamax.hidden_markov_model.models.abstractions import HMMInitialState
from dynamax.parameters import ParameterProperties
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from typing import NamedTuple, Union


class ParamsStandardHMMInitialState(NamedTuple):
    probs: Union[Float[Array, "state_dim"], ParameterProperties]


class StandardHMMInitialState(HMMInitialState):
    """Abstract class for HMM initial distributions.
    """
    def __init__(self,
                 num_states,
                 initial_probs_concentration=1.1):
        """
        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
        """
        self.num_states = num_states
        self.initial_probs_concentration = initial_probs_concentration * jnp.ones(num_states)

    def distribution(self, params, inputs=None):
        return tfd.Categorical(probs=params.probs)

    def initialize(self, key=None, method="prior", initial_probs=None):
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
            this_key, key = jr.split(key)
            initial_probs = tfd.Dirichlet(self.initial_probs_concentration).sample(seed=this_key)

        # Package the results into dictionaries
        params = ParamsStandardHMMInitialState(probs=initial_probs)
        props = ParamsStandardHMMInitialState(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params):
        return tfd.Dirichlet(self.initial_probs_concentration).log_prob(params.probs)

    def _compute_initial_probs(self, params, inputs=None):
        return params.probs

    def collect_suff_stats(self, params, posterior, inputs=None):
        return posterior.smoothed_probs[0]

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.probs.trainable:
            if self.num_states == 1:
                probs = jnp.array([1.0])
            else:
                expected_initial_counts = batch_stats.sum(axis=0)
                probs = tfd.Dirichlet(self.initial_probs_concentration + expected_initial_counts).mode()
            params = params._replace(probs=probs)
        return params, m_step_state

