from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class PoissonHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_rates,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_rates_prior_concentration=1.1,
                 emission_rates_prior_rate=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self._emission_rates = Parameter(emission_rates, bijector=tfb.Invert(tfb.Softplus()))
        self._emission_rates_prior_concentration = Parameter(emission_rates_prior_concentration,
                                                             is_frozen=True,
                                                             bijector=tfb.Invert(tfb.Softplus()))
        self._emission_rates_prior_rate = Parameter(emission_rates_prior_rate,
                                                    is_frozen=True,
                                                    bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_rates = jr.exponential(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_rates)

    @property
    def emission_rates(self):
        return self._emission_rates

    # Properties to get various parameters of the model
    def emission_distribution(self, state):
        return tfd.Independent(tfd.Poisson(rate=self.emission_rates.value[state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
        lp += tfd.Gamma(self._emission_rates_prior_concentration.value,
                          self._emission_rates_prior_rate.value).log_prob(self._emission_rates.value).sum()
        return lp

    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class PoissonHMMSuffStats:
            # Wrapper for sufficient statistics of a BernoulliHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_w: chex.Array
            sum_x: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self.initial_probs.value,
                                     self.transition_matrix.value,
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)[:, None]
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)

            # Pack into a dataclass
            stats = PoissonHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=initial_probs,
                trans_probs=trans_probs,
                sum_w=sum_w,
                sum_x=sum_x,
            )
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # Then maximize the expected log probability as a fn of model parameters
        post_concentration = self._emission_rates_prior_concentration.value + stats.sum_x
        post_rate = self._emission_rates_prior_rate.value + stats.sum_w
        self._emission_rates.value = tfd.Gamma(post_concentration, post_rate).mode()
