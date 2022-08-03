import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class MultinomialHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_probs,
                 num_trials=1,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        # Check shapes
        assert emission_probs.ndim == 3, \
            "emission_probs must be (num_states x num_emissions x num_classes)"
        num_classes = emission_probs.shape[2]
        self._num_trials = num_trials
        self._emission_probs = Parameter(emission_probs, bijector=tfb.Invert(tfb.SoftmaxCentered()))
        self._emission_prior_concentration = Parameter(emission_prior_concentration  * jnp.ones(num_classes),
                                                       is_frozen=True,
                                                       bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, num_emissions, num_classes, num_trials=1):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.dirichlet(key3, jnp.ones(num_classes), (num_states, num_emissions))
        return cls(initial_probs, transition_matrix, emission_probs, num_trials=num_trials)

    @property
    def emission_probs(self):
        return self._emission_probs

    @property
    def num_emissions(self):
        return self.emission_probs.shape[1]

    @property
    def num_classes(self):
        return self.emission_probs.shape[2]

    @property
    def num_trials(self):
        return self._num_trials

    def emission_distribution(self, state):
        return tfd.Independent(tfd.Multinomial(self._num_trials,
                                               probs=self.emission_probs.value[state]),
                               retinterpreted_batch_ndims=1)

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
        lp += tfd.Dirichlet(self._emission_prior_concentration.value).log_prob(self._emission_probs.value).sum()
        return lp
