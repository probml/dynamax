import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.parameters import ParameterProperties
from jaxtyping import Float, Array
from typing import NamedTuple, Union


class ParamsStandardHMMTransitions(NamedTuple):
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
    def __init__(self, num_states, concentration=1.1, stickiness=0.0):
        """
        Args:
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        self.num_states = num_states
        self.concentration = \
            concentration * jnp.ones((num_states, num_states)) + \
                stickiness * jnp.eye(num_states)

    def distribution(self, params, state, inputs=None):
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(self, key=None, method="prior", transition_matrix=None):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            transition_matrix (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if transition_matrix is None:
            this_key, key = jr.split(key)
            transition_matrix = tfd.Dirichlet(self.concentration).sample(seed=this_key)

        # Package the results into dictionaries
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params):
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def _compute_transition_matrices(self, params, inputs=None):
        return params.transition_matrix

    def collect_suff_stats(self, params, posterior, inputs=None):
        return posterior.trans_probs

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats.sum(axis=0)
                transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state
