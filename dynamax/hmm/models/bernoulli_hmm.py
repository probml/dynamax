import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


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

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def emission_distribution(self, params, state, covariates=None):
        # This model assumes the emissions are a vector of conditionally independent
        # Bernoulli observations. The `reinterpreted_batch_ndims` argument tells
        # `tfd.Independent` that only the last dimension should be considered a "batch"
        # of conditionally independent observations.
        return tfd.Independent(tfd.Bernoulli(probs=params['emissions']['probs'][state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Beta(self.emission_prior_concentration1, self.emission_prior_concentration0).log_prob(
            params['emissions']['probs']).sum()
        return lp

    def _initialize_emissions(self, key):
        emission_probs = jr.uniform(key, (self.num_states, self.emission_dim))
        params = dict(probs=emission_probs)
        param_props = dict(probs=ParameterProperties(constrainer=tfb.Sigmoid()))
        return  params, param_props

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        sum_x = jnp.zeros((self.num_states, self.emission_dim)),
        sum_1mx = jnp.zeros((self.num_states, self.emission_dim)),
        return (sum_x, sum_1mx)

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_x = jnp.einsum("tk, ti->ki", expected_states, jnp.where(jnp.isnan(emissions), 0, emissions))
        sum_1mx = jnp.einsum("tk, ti->ki", expected_states,
                                jnp.where(jnp.isnan(emissions), 0, 1 - emissions))
        return (sum_x, sum_1mx)

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['probs'].trainable:
            sum_x, sum_1mx = emission_stats
            params['emissions']['probs'] = tfd.Beta(
                self.emission_prior_concentration1 + sum_x,
                self.emission_prior_concentration0 + sum_1mx).mode()
        return params
