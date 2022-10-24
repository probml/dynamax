import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


class PoissonHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1,
                 emission_prior_rate=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_concentration = emission_prior_concentration
        self.emission_prior_rate = emission_prior_rate

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        emission_rates = jr.exponential(key, (self.num_states, self.emission_dim))
        params = dict(rates=emission_rates)
        param_props = dict(rates=ParameterProperties(constrainer=tfb.Softplus()))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.Independent(tfd.Poisson(rate=params['emissions']['rates'][state]),
                               reinterpreted_batch_ndims=1)

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        return dict(
            sum_w=jnp.zeros((self.num_states, 1)),
            sum_x=jnp.zeros((self.num_states, self.emission_dim)),
        )

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Gamma(self.emission_prior_concentration, self.emission_prior_rate).log_prob(
            params['emissions']['rates']).sum()
        return lp

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_w = jnp.einsum("tk->k", expected_states)[:, None]
        sum_x = jnp.einsum("tk, ti->ki", expected_states, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x)

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['rates'].trainable:
            post_concentration = self.emission_prior_concentration + emission_stats['sum_x']
            post_rate = self.emission_prior_rate + emission_stats['sum_w']
            params['emissions']['rates'] = tfd.Gamma(post_concentration, post_rate).mode()
        return params
