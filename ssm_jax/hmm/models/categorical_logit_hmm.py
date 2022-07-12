from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.inference import _get_batch_emission_probs
from ssm_jax.hmm.models.categorical_hmm import CategoricalHMM


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
        self._initial_distribution = tfd.Categorical(logits=initial_logits)
        self._transition_distribution = tfd.Categorical(logits=transition_logits)
        self._emission_distribution = tfd.Categorical(logits=emission_logits)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def initial_probabilities(self):
        return self._initial_distribution.probs_parameter()

    @property
    def emission_probs(self):
        return self._emission_distribution.probs_parameter()

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs_parameter()

    @property
    def initial_logits(self):
        return self._initial_distribution.logits_parameter()

    @property
    def transition_logits(self):
        return self._transition_distribution.logits_parameter()

    @property
    def emission_logits(self):
        return self.emission_distribution.logits_parameter()

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.normal(key1, (num_states,))
        transition_matrix = jr.normal(key2, (num_states, num_states))
        emission_probs = jr.normal(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (self.initial_logits, self.transition_logits, self.emission_logits)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)

    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(0.01), num_iters=50):

        partial_get_emission_probs = partial(_get_batch_emission_probs, self)
        batch_emission_probs = vmap(partial_get_emission_probs)(batch_emissions, batch_posteriors.smoothed_probs)

        emission_probs = batch_emission_probs.sum(axis=0)
        denom = emission_probs.sum(axis=-1, keepdims=True)
        emission_logits = jnp.log(emission_probs / jnp.where(denom == 0, 1, denom))

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transition_logits = jnp.log(transitions_probs / jnp.where(denom == 0, 1, denom))
        batch_initial_probs = batch_posteriors.smoothed_probs[:, 0, :]
        initial_logits = jnp.log(batch_initial_probs.sum(axis=0) / batch_initial_probs.sum())

        hmm = CategoricalLogitHMM(initial_logits, transition_logits, emission_logits)

        return hmm, batch_posteriors.marginal_loglik
