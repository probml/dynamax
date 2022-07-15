import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.models.base import BaseHMM
from ssm_jax.hmm.models.utils import get_training_parametrization


@register_pytree_node_class
class MultinomialHMM(BaseHMM):
    def __init__(self, initial_probabilities, transition_matrix, emission_probs, num_trials=1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._num_trials = num_trials
        self._num_trials = num_trials
        self._emission_probs = emission_probs

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.uniform(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    def emission_distribution(self, state):
        return tfd.Multinomial(self._num_trials, probs=self._emission_probs[state])

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

    @property
    def num_trials(self):
        return self._num_trials

    def training_parametrization(self, params_names="ite"):
        initial_dist_params = self._initial_probabilities
        transition_dist_params = self._transition_matrix
        emission_dist_params = (self._emission_probs,)
        return get_training_parametrization(initial_dist_params, transition_dist_params, emission_dist_params,
                                            self.hyperparams, self.unconstrained_params, self.from_unconstrained_params,
                                            params_names)