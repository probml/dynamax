from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.abstractions import Parameter
from ssm_jax.distributions import NormalInverseWishart
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import ExponentialFamilyHMM
from ssm_jax.utils import PSDToRealBijector


@chex.dataclass
class MultivariateNormalTiedHMMSuffStats:
    # Wrapper for sufficient statistics of a GaussianHMM
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    sum_w: chex.Array
    sum_x: chex.Array
    sum_xxT: chex.Array


@register_pytree_node_class
class MultivariateNormalTiedHMM(ExponentialFamilyHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrix,
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
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self._emission_means = Parameter(emission_means)
        self._emission_cov = Parameter(emission_covariance_matrix, bijector=PSDToRealBijector)

        dim = emission_means.shape[-1]
        self._emission_prior_mean = Parameter(emission_prior_mean * jnp.ones(dim), is_frozen=True)
        self._emission_prior_conc = Parameter(emission_prior_concentration,
                                              is_frozen=True,
                                              bijector=tfb.Invert(tfb.Softplus()))
        self._emission_prior_scale = Parameter(
            emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 else emission_prior_scale * jnp.eye(dim),
            is_frozen=True,
            bijector=PSDToRealBijector)
        self._emission_prior_df = Parameter(dim + emission_prior_extra_df,
                                            is_frozen=True,
                                            bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_cov = jnp.eye(emission_dim)
        return cls(initial_probs, transition_matrix, emission_means, emission_cov)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self._emission_means

    @property
    def emission_covariance_matrix(self):
        return self._emission_cov

    def emission_distribution(self, state):
        return tfd.MultivariateNormalFullCovariance(self._emission_means.value[state], self._emission_cov.value)

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

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()

        lp += NormalInverseWishart(self._emission_prior_mean.value, self._emission_prior_conc.value,
                                   self._emission_prior_df.value, self._emission_prior_scale.value).log_prob(
                                       (self.emission_covariance_matrix.value, self.emission_means.value)).sum()
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

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = MultivariateNormalTiedHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                                       initial_probs=initial_probs,
                                                       trans_probs=trans_probs,
                                                       sum_w=sum_w,
                                                       sum_x=sum_x,
                                                       sum_xxT=sum_xxT)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)
        self.emission_means.value = stats.sum_x / stats.sum_w[:, None]
        self.emission_covariance_matrix.value = 1 / stats.sum_w.sum() * (
            stats.sum_xxT - jnp.einsum("ki,kj->kij", stats.sum_x, stats.sum_x) / stats.sum_w[:, None, None]).sum(axis=0)
