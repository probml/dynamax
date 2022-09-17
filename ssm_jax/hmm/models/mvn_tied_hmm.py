from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import tree_map
from ssm_jax.parameters import ParameterProperties
from ssm_jax.distributions import InverseWishart
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import ExponentialFamilyHMM
from ssm_jax.utils import PSDToRealBijector


@chex.dataclass
class MultivariateNormalTiedHMMSuffStats:
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    sum_w: chex.Array
    sum_x: chex.Array
    sum_xxT: chex.Array


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

    def random_initialization(self, key):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_means = jr.normal(key3, (self.num_states, self.emission_dim))
        emission_cov = jnp.eye(self.emission_dim)

        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(means=emission_means, covariance=emission_cov))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), covariance=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))))
        return  params, param_props

    def emission_distribution(self, params, state):
        return tfd.MultivariateNormalFullCovariance(
            params['emissions']['means'][state], params['emissions']['cov'])

    @property
    def suff_stats_event_shape(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        return MultivariateNormalTiedHMMSuffStats(
            marginal_loglik=(),
            initial_probs=(self.num_states,),
            trans_probs=(self.num_states, self.num_states),
            sum_w=(self.num_states,),
            sum_x=(self.num_states, self.num_obs),
            sum_xxT=(self.num_obs, self.num_obs),
        )

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        lp += InverseWishart(self.emission_prior_df, self.emission_prior_scale).log_prob(
            params['emissions']['cov'])
        lp += tfd.MultivariateNormalFullCovariance(
            self.emission_prior_mean, self.emission_prior_conc * params['emissions']['cov']).log_prob(
            params['emissions']['means']).sum()
        return lp

    def _zeros_like_suff_stats(self):
        dim = self.num_obs
        num_states = self.num_states
        return MultivariateNormalTiedHMMSuffStats(
            marginal_loglik=0.0,
            initial_probs=jnp.zeros((num_states,)),
            trans_probs=jnp.zeros((num_states, num_states)),
            sum_w=jnp.zeros((num_states,)),
            sum_x=jnp.zeros((num_states, dim)),
            sum_xxT=jnp.zeros((num_states, dim, dim)),
        )

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(), self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk,ti->ki", posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, emissions, emissions)
            stats = MultivariateNormalTiedHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                                       initial_probs=initial_probs,
                                                       trans_probs=trans_probs,
                                                       sum_w=sum_w,
                                                       sum_x=sum_x,
                                                       sum_xxT=sum_xxT)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, param_props, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)
        params['emissions']['means'] = stats.sum_x / stats.sum_w[:, None]
        params['emissions']['cov'] = 1 / stats.sum_w.sum() * (
            stats.sum_xxT - jnp.einsum("ki,kj->kij", stats.sum_x, stats.sum_x) / stats.sum_w[:, None, None]).sum(axis=0)
