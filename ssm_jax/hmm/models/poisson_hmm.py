import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import nn
from jax import tree_map
from jax import vmap
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
        self._emission_log_rates = emission_log_rates

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_log_rates = jnp.log(jr.exponential(key3, (num_states, emission_dim)))
        return cls(initial_probs, transition_matrix, emission_log_rates)

    # Properties to get various parameters of the model
    def emission_distribution(self, state):
        return tfd.Independent(tfd.Poisson(log_rate=self._emission_log_rates[state]), reinterpreted_batch_ndims=1)

    @property
    def emission_rates(self):
        return jnp.exp(self._emission_log_rates)

    @property
    def emission_log_rates(self):
        return self._emission_log_rates

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            nn.softmax(jnp.log(self.initial_probabilities), axis=-1),
            nn.softmax(jnp.log(self.transition_matrix), axis=-1),
            self.emission_log_rates,
        )

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)

    def _sufficient_statistics(self, datapoint):
        return (datapoint, jnp.ones_like(datapoint))

    def m_step(self, batch_emissions, batch_posteriors, **kwargs):

        # TODO: This naming needs to be fixed up by changing BaseHMM.e_step
        batch_posteriors, batch_trans_probs = batch_posteriors

        def flatten(x):
            return x.reshape(-1, x.shape[-1])

        # TODO: This should use smoothed_probs
        filtered_probs = batch_posteriors.filtered_probs
        flat_weights = flatten(filtered_probs)
        flat_data = flatten(batch_emissions)

        stats = vmap(self._sufficient_statistics)(flat_data)
        stats = tree_map(lambda x: jnp.einsum("nk,n...->k...", flat_weights, x), stats)

        concentration, rate = stats
        emission_rates = tfd.Gamma(concentration, rate).mode()
        emission_log_rates = jnp.log(emission_rates)

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transitions_probs = transitions_probs / jnp.where(denom == 0, 1, denom)

        batch_initial_probs = filtered_probs[:, 0, :]
        initial_probs = batch_initial_probs.sum(axis=0) / batch_initial_probs.sum()

        hmm = PoissonHMM(initial_probs, transitions_probs, emission_log_rates)

        return hmm
