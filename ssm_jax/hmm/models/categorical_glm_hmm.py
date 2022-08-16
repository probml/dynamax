from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.nn import one_hot
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class CategoricalRegressionHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_matrices,
                 emission_biases,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        # Save parameters and hyperparameters
        self._num_classes = emission_biases.shape[0]
        self._emission_matrices = Parameter(emission_matrices)
        self._emission_biases = Parameter(emission_biases)
        self._emission_prior_concentration = Parameter(emission_prior_concentration * jnp.ones(self._num_classes),
                                                       is_frozen=True,
                                                       bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, num_classes, feature_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_matrices = jr.normal(key3, (num_states, num_classes, feature_dim))
        emission_biases = jr.normal(key4, (num_states, num_classes))
        return cls(initial_probs, transition_matrix, emission_matrices, emission_biases)

    # Properties to get various parameters of the model
    @property
    def emission_matrices(self):
        return self._emission_matrices

    @property
    def emission_biases(self):
        return self._emission_biases

    @property
    def num_classes(self):
        return self.emission_probs.value.shape[2]

    def emission_distribution(self, state, **covariates):
        logits = self._emission_matrices.value[state] @ covariates['features'] + self._emission_biases.value[state]
        return tfd.Categorical(logits=logits)

    @property
    def emission_distribution_parameters(self):
        return dict(emission_matrices=self._emission_matrices, emission_biases=self._emission_biases)
