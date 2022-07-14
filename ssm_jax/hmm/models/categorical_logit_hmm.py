from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jax import vmap
from jax import tree_map
from jax.tree_util import register_pytree_node_class

import chex
import tensorflow_probability.substrates.jax.distributions as tfd

from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.models.categorical_hmm import CategoricalHMM
from ssm_jax.utils import one_hot


@register_pytree_node_class
class CategoricalLogitHMM(CategoricalHMM):
    def __init__(self, initial_logits, transition_logits, emission_logits):
        num_states, num_emissions = emission_logits.shape

        # Check shapes
        assert initial_logits.shape == (num_states,)
        assert transition_logits.shape == (num_states, num_states)

        self._num_states = num_states
        self._num_emissions = num_emissions

        # Construct the  distribution objects
        self._initial_logits = initial_logits
        self._transition_logits = transition_logits
        self._emission_logits = emission_logits

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_logits = jr.normal(key1, (num_states,))
        transition_logits = jr.normal(key2, (num_states, num_states))
        emission_logits = jr.normal(key3, (num_states, emission_dim))
        return cls(initial_logits, transition_logits, emission_logits)

    def initial_distribution(self):
        return tfd.Categorical(logits=self._initial_logits)

    @property
    def initial_probabilities(self):
        logits = self._initial_logits
        return jnp.exp(logits - logsumexp(logits, keepdims=True))

    def transition_distribution(self, state):
        return tfd.Categorical(logits=self._transition_logits[state])

    @property
    def transition_matrix(self):
        logits = self._transition_logits
        return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def emission_distribution(self, state):
        return tfd.Categorical(logits=self._emission_logits[state])

    @property
    def emission_probs(self):
        logits = self._emission_logits
        return jnp.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (self._initial_logits, self._transition_logits, self._emission_logits)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)

    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """
        @chex.dataclass
        class CategoricalHMMSuffStats:
            # Wrapper for sufficient statistics of a BernoulliHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_x: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self.initial_probabilities,
                                     self.transition_matrix,
                                     self._conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Compute the expected sufficient statistics
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs,
                               one_hot(emissions, self.num_states))
            
            # Pack into a dataclass
            stats = CategoricalHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=initial_probs,
                trans_probs=trans_probs,
                sum_x=sum_x,
            )
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    @classmethod
    def m_step(cls, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # Manually compute the logits for the initial, transition, and emission dist
        initial_logits = jnp.log(stats.initial_probs + 1e-1)
        transition_logits = jnp.log(stats.trans_probs + 1e-1)
        emission_logits = jnp.log(stats.sum_x + 1e-1)
        return cls(initial_logits, transition_logits, emission_logits)
