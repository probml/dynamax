import jax.numpy as jnp
import jax.random as jr
from jax.nn import one_hot
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


class CategoricalHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 num_emissions,
                 num_classes,
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
        self.num_emissions = num_emissions
        self.num_classes = num_classes
        self.emission_prior_concentration = emission_prior_concentration  * jnp.ones(num_classes)

    @property
    def emission_shape(self):
        return (self.num_emissions,)

    def emission_distribution(self, params, state, covariates=None):
        return tfd.Independent(
            tfd.Categorical(probs=params['emissions']['probs'][state]),
            reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Dirichlet(self.emission_prior_concentration).log_prob(
            params['emissions']['probs']).sum()
        return lp

    def _initialize_emissions(self, key):
        emission_probs = jr.dirichlet(key, jnp.ones(self.num_classes), (self.num_states, self.num_emissions))
        params = dict(probs=emission_probs)
        param_props = dict(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return  params, param_props

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        return dict(sum_x=jnp.zeros((self.num_states, self.num_obs, self.num_classes)))

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['probs'].trainable:
            params['emissions']['probs'] = tfd.Dirichlet(
                self.emission_prior_concentration + emission_stats['sum_x']).mode()
        return params
