import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.tree_util import register_pytree_node_class
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

        self._emission_probs = emission_probs

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.uniform(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    def emission_distribution(self, state):
        return tfd.Independent(tfd.Bernoulli(probs=self._emission_probs[state]),
                               reinterpreted_batch_ndims=1)

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            tfb.SoftmaxCentered().inverse(self.initial_probabilities),
            tfb.SoftmaxCentered().inverse(self.transition_matrix),
            tfb.Sigmoid().inverse(self.emission_probs),
        )

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.Sigmoid().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)

    def _sufficient_statistics(self, datapoint):
        return datapoint, 1 - datapoint

    def m_step(self, batch_emissions, batch_posteriors, **kwargs):
        """
        Another  way to calculate emission probs:

        smoothed_probs = batch_posteriors.smoothed_probs

        def get_expected_probs(x, y):
            return x.reshape((-1, 1)) * jnp.tile(y, reps=(self.num_states, 1))

        emission_probs1 = vmap(lambda x, y: vmap(get_expected_probs)(x, y))(smoothed_probs, batch_emissions)
        emission_probs0 = vmap(lambda x, y: vmap(get_expected_probs)(x, y))(smoothed_probs, 1 - batch_emissions)

        emission_probs1 = jnp.sum(emission_probs1, axis=0)
        emission_probs1 = jnp.sum(emission_probs1, axis=0)

        emission_probs0 = jnp.sum(emission_probs0, axis=0)
        emission_probs0 = jnp.sum(emission_probs0, axis=0)
        emission_probs = emission_probs1 / (emission_probs1 + emission_probs0)

        """
        # TODO: This naming needs to be fixed up by changing BaseHMM.e_step
        batch_posteriors, batch_trans_probs = batch_posteriors

        def flatten(x):
            return x.reshape(-1, x.shape[-1])

        smoothed_probs = batch_posteriors.smoothed_probs
        flat_weights = flatten(smoothed_probs)
        flat_data = flatten(batch_emissions)

        stats = vmap(self._sufficient_statistics)(flat_data)
        stats = tree_map(lambda x: jnp.einsum("nk,n...->k...", flat_weights, x), stats)

        prior = tfd.Beta(1.1, 1.1)
        stats = tree_map(jnp.add, stats, (prior.concentration1, prior.concentration0))
        concentration1, concentration0 = stats
        emission_probs = tfd.Beta(concentration1, concentration0).mode()

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transitions_probs = transitions_probs / jnp.where(denom == 0, 1, denom)

        batch_initial_probs = smoothed_probs[:, 0, :]
        initial_probs = batch_initial_probs.sum(axis=0) / batch_initial_probs.sum()

        hmm = BernoulliHMM(initial_probs, transitions_probs, emission_probs)

        return hmm
