import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


class MultinomialHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_classes,
                 num_trials,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
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
        self.num_classes = num_classes
        self.num_trials = num_trials
        self.emission_prior_concentration = emission_prior_concentration * jnp.ones(num_classes)

    @property
    def emission_shape(self):
        return (self.emission_dim, self.num_classes)

    def _initialize_emissions(self, key):
        emission_probs = jr.dirichlet(key, jnp.ones(self.num_classes), (self.num_states, self.emission_dim))
        params = dict(probs=emission_probs)
        param_props = dict(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.Independent(tfd.Multinomial(self.num_trials, probs=params['emissions']['probs'][state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Dirichlet(self.emission_prior_concentration).log_prob(
            params['emissions']['probs']).sum()
        return lp

    def _zeros_like_suff_stats(self):
        return dict(sum_x=jnp.zeros((self.num_states, self.emission_dim, self.num_classes)))

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        return dict(sum_x=jnp.einsum("tk, tdi->kdi", expected_states, emissions))

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['probs'].trainable:
            params['emissions']['probs'] = tfd.Dirichlet(
                self.emission_prior_concentration + emission_stats['sum_x']).mode()
        return params
