from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.inference import _get_batch_emission_probs
from ssm_jax.hmm.models.base import BaseHMM


@register_pytree_node_class
class CategoricalHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_probs):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        num_states, num_emissions = emission_probs.shape

        # Check shapes
        assert initial_probabilities.shape == (num_states,)
        assert transition_matrix.shape == (num_states, num_states)

        self._num_states = num_states
        self._num_emissions = num_emissions

        self._emission_distribution = tfd.Categorical(probs=emission_probs)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.dirichlet(key3, jnp.ones(emission_dim), (num_states,))
        return cls(initial_probs, transition_matrix, emission_probs)

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix),
                tfb.SoftmaxCentered().inverse(self.emission_probs))

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.SoftmaxCentered().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)

    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(0.01), num_iters=50):
        partial_get_emission_probs = partial(_get_batch_emission_probs, self)
        batch_emission_probs = vmap(partial_get_emission_probs)(batch_emissions, batch_posteriors.smoothed_probs)

        emission_probs = batch_emission_probs.sum(axis=0)
        denom = emission_probs.sum(axis=-1, keepdims=True)
        emission_probs = emission_probs / jnp.where(denom == 0, 1, denom)

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transition_probs = transitions_probs / jnp.where(denom == 0, 1, denom)

        batch_initial_probs = batch_posteriors.smoothed_probs[:, 0, :]
        initial_probs = batch_initial_probs.sum(axis=0) / batch_initial_probs.sum()

        hmm = CategoricalHMM(initial_probs, transition_probs, emission_probs)
        return hmm, batch_posteriors.marginal_loglik
