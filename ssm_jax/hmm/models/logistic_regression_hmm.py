import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class LogisticRegressionHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_matrices,
                 emission_biases,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        # Save parameters and hyperparameters
        self._emission_matrices = Parameter(emission_matrices)
        self._emission_biases = Parameter(emission_biases)

    @classmethod
    def random_initialization(cls, key, num_states, feature_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_matrices = jr.normal(key3, (num_states, feature_dim))
        emission_biases = jr.normal(key4, (num_states,))
        return cls(initial_probs, transition_matrix, emission_matrices, emission_biases)

    # Properties to get various parameters of the model
    @property
    def emission_matrices(self):
        return self._emission_matrices

    @property
    def emission_biases(self):
        return self._emission_biases

    def emission_distribution(self, state, **covariates):
        logits = self._emission_matrices.value[state] @ covariates['features'] + self._emission_biases.value[state]
        return tfd.Bernoulli(logits=logits)