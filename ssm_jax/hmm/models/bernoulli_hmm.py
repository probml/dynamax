from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from ssm_jax.parameters import ParameterProperties
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import ExponentialFamilyHMM

@chex.dataclass
class BernoulliHMMSuffStats:
    # Wrapper for sufficient statistics of a BernoulliHMM
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    sum_x: chex.Array
    sum_1mx: chex.Array


class BernoulliHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration1=1.1,
                 emission_prior_concentration0=1.1):
        """_summary_
        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_dim
        self.emission_prior_concentration0 = emission_prior_concentration0
        self.emission_prior_concentration1 = emission_prior_concentration1

    def random_initialization(self, key):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_probs = jr.uniform(key3, (self.num_states, self.emission_dim))
        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(probs=emission_probs))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Sotfplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(probs=ParameterProperties(constrainer=tfb.Sigmoid())))
        return  params, param_props

    def emission_distribution(self, params, state):
        return tfd.Independent(tfd.Bernoulli(probs=params['emissions']['probs'][state]),
                               reinterpreted_batch_ndims=1)

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        return BernoulliHMMSuffStats(
            marginal_loglik = 0.0,
            initial_probs   = jnp.zeros((self.num_states,)),
            trans_probs     = jnp.zeros((self.num_states, self.num_states)),
            sum_x           = jnp.zeros((self.num_states, self.num_obs)),
            sum_1mx         = jnp.zeros((self.num_states, self.num_obs)),
        )

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration.value).log_prob(
            self.initial_probs.value)
        lp += tfd.Dirichlet(self.transition_matrix_concentration.value).log_prob(
            self.transition_matrix.value).sum()
        lp += tfd.Beta(self.emission_prior_concentration1.value,
                       self.emission_prior_concentration0.value).log_prob(
                           params['emissions']['probs']).sum()
        return lp

    def e_step(self, params, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(params),
                                     self._compute_transition_matrices(params),
                                     self._compute_conditional_logliks(params, emissions))

            # Compute the initial state and transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, jnp.where(jnp.isnan(emissions), 0, emissions))
            sum_1mx = jnp.einsum("tk, ti->ki", posterior.smoothed_probs,
                                 jnp.where(jnp.isnan(emissions), 0, 1 - emissions))

            # Pack into a dataclass
            stats = BernoulliHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                          initial_probs=posterior.initial_probs,
                                          trans_probs=trans_probs,
                                          sum_x=sum_x,
                                          sum_1mx=sum_1mx)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # Then maximize the expected log probability as a fn of model parameters
        params['emissions']['probs'] = tfd.Beta(
            self._emission_prior_concentration1.value + stats.sum_x,
            self._emission_prior_concentration0.value + stats.sum_1mx).mode()

        return params
