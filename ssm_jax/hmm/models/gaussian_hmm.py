from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import BaseHMM
from ssm_jax.utils import PSDToRealBijector


@register_pytree_node_class
class GaussianHMM(BaseHMM):
    def __init__(self, initial_probabilities, transition_matrix, emission_means, emission_covariance_matrices):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_distribution = tfd.MultivariateNormalFullCovariance(emission_means, emission_covariance_matrices)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_means(self):
        return self._emission_distribution.mean()

    @property
    def emission_covariance_matrices(self):
        return self._emission_distribution.covariance()

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            tfb.SoftmaxCentered().inverse(self.initial_probabilities),
            tfb.SoftmaxCentered().inverse(self.transition_matrix),
            self.emission_means,
            PSDToRealBijector.forward(self.emission_covariance_matrices),
        )

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_means = unconstrained_params[2]
        emission_covs = PSDToRealBijector.inverse(unconstrained_params[3])
        return cls(initial_probabilities, transition_matrix, emission_means, emission_covs, *hypers)

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class GaussianHMMSuffStats:
            # Wrapper for sufficient statistics of a GaussianHMM
            initial_probs: chex.Array
            sum_trans_probs: chex.Array
            sum_w: chex.Array
            sum_x: chex.Array
            sum_xxT: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(
                self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions)
            )

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            sum_trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, emissions, emissions)

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = GaussianHMMSuffStats(
                initial_probs=initial_probs, sum_trans_probs=sum_trans_probs, sum_w=sum_w, sum_x=sum_x, sum_xxT=sum_xxT
            )
            return stats, posterior.marginal_loglik

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    @classmethod
    def m_step(cls, batch_stats):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)

        # Initial distribution
        initial_probs = tfd.Dirichlet(1.0001 + stats.initial_probs).mode()

        # Transition distribution
        transition_matrix = tfd.Dirichlet(1.0001 + stats.sum_trans_probs).mode()

        # Gaussian emission distribution
        emission_dim = stats.sum_x.shape[-1]
        emission_means = stats.sum_x / stats.sum_w[:, None]
        emission_covs = (
            stats.sum_xxT / stats.sum_w[:, None, None]
            - jnp.einsum("ki,kj->kij", emission_means, emission_means)
            + 1e-4 * jnp.eye(emission_dim)
        )

        # Pack the results into a new GaussianHMM
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)
