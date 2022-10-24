import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import tree_map
from dynamax.parameters import ParameterProperties
from dynamax.distributions import InverseWishart
from dynamax.hmm.models.base import ExponentialFamilyHMM
from dynamax.utils import PSDToRealBijector


class MultivariateNormalTiedHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrix (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
            else emission_prior_scale * jnp.eye(emission_dim)
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        emission_means = jr.normal(key, (self.num_states, self.emission_dim))
        emission_cov = jnp.eye(self.emission_dim)

        params = dict(means=emission_means, cov=emission_cov)
        param_props = dict(means=ParameterProperties(), cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalFullCovariance(
            params['emissions']['means'][state], params['emissions']['cov'])

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += InverseWishart(self.emission_prior_df, self.emission_prior_scale).log_prob(
            params['emissions']['cov'])
        lp += tfd.MultivariateNormalFullCovariance(
            self.emission_prior_mean, self.emission_prior_conc * params['emissions']['cov']).log_prob(
            params['emissions']['means']).sum()
        return lp

    def _zeros_like_suff_stats(self):
        dim = self.num_obs
        num_states = self.num_states
        return dict(
            sum_w=jnp.zeros((num_states,)),
            sum_x=jnp.zeros((num_states, dim)),
            sum_xxT=jnp.zeros((num_states, dim, dim)),
        )

    # Expectation-maximization (EM) code
    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        stats = dict(sum_w=sum_w, sum_x=sum_x, sum_xxT=sum_xxT)
        return stats

    def _m_step_emissions(self, params, param_props, emission_stats):
        sum_w = emission_stats['sum_w']
        sum_x = emission_stats['sum_x']
        sum_xxT = emission_stats['sum_xxT']
        params['emissions']['means'] = sum_x / sum_w[:, None]
        params['emissions']['cov'] = (1 / sum_w.sum()) * (
            sum_xxT - jnp.einsum("ki,kj->kij", sum_x, sum_x) / sum_w[:, None, None]).sum(axis=0)
        return params
