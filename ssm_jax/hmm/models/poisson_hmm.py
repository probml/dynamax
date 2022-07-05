import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import nn
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.models.base import BaseHMM

# Using TFP for now since it has all our distributions
# (Distrax doesn't have Poisson, it seems.)


@register_pytree_node_class
class PoissonHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_log_rates):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)
        self._emission_distribution = tfd.Poisson(log_rate=emission_log_rates)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_log_rates = jnp.log(jr.exponential(key3, (num_states, emission_dim)))
        return cls(initial_probs, transition_matrix, emission_log_rates)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_rates(self):
        return jnp.exp(self.emission_log_rates)

    @property
    def emission_log_rates(self):
        return self._emission_distribution.log_rate

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (nn.softmax(jnp.log(self.initial_probabilities),
                           axis=-1), nn.softmax(jnp.log(self.transition_matrix), axis=-1), self.emission_log_rates)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)
