from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import BaseHMM


@register_pytree_node_class
class BernoulliHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_probs):
        """_summary_
        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_probs = Parameter(emission_probs, bijector=tfb.Invert(tfb.Sigmoid()))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.uniform(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    @property
    def emission_probs(self):
        return self._emission_probs

    def emission_distribution(self, state):
        return tfd.Independent(tfd.Bernoulli(probs=self._emission_probs.value[state]),
                               reinterpreted_batch_ndims=1)

    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class BernoulliHMMSuffStats:
            # Wrapper for sufficient statistics of a BernoulliHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_x: chex.Array
            sum_1mx: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self.initial_probs.value,
                                     self.transition_matrix.value,
                                     self._conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, jnp.where(jnp.isnan(emissions), 0, emissions))
            sum_1mx = jnp.einsum("tk, ti->ki", posterior.smoothed_probs,
                                 jnp.where(jnp.isnan(emissions), 0, 1 - emissions))

            # Pack into a dataclass
            stats = BernoulliHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                          initial_probs=initial_probs,
                                          trans_probs=trans_probs,
                                          sum_x=sum_x,
                                          sum_1mx=sum_1mx)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def m_step(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)
        # Then maximize the expected log probability as a fn of model parameters
        self._initial_probs.value = tfd.Dirichlet(1.0001 + stats.initial_probs).mode()
        self._transition_matrix.value = tfd.Dirichlet(1.0001 + stats.trans_probs).mode()
        self._emission_probs.value = tfd.Beta(1.1 + stats.sum_x, 1.1 + stats.sum_1mx).mode()
