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

        self._emission_means = emission_means
        self._emission_covs = emission_covariance_matrices

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
    def emission_means(self):
        return self._emission_means

    @property
    def emission_covariance_matrices(self):
        return self._emission_covs

    def emission_distribution(self, state):
        return tfd.MultivariateNormalFullCovariance(
            self._emission_means[state], self._emission_covs[state])

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """
        @chex.dataclass
        class NormalizedGaussianHMMSuffStats:
            # Wrapper for (normalized) sufficient statistics of a GaussianHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_w: chex.Array
            normd_x: chex.Array
            normd_xxT: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(
                self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions)
            )

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Compute the normalized expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            normd_w = posterior.smoothed_probs / sum_w
            normd_x = jnp.einsum("tk, ti->ki", normd_w, emissions)
            normd_xxT = jnp.einsum("tk, ti, tj->kij", normd_w, emissions, emissions)

            # Catch NaNs that result from DBZ due to sum_w = 0
            emissions_dim = emissions.shape[-1]
            normd_x = jnp.where(sum_w[:, None] > 1e-6,
                                normd_x,
                                jnp.zeros(emissions_dim))
            normd_xxT = jnp.where(sum_w[:, None, None] > 1e-6,
                                  normd_xxT,
                                  1e6 * jnp.eye(emissions_dim))

            stats = NormalizedGaussianHMMSuffStats(
                marginal_loglik=posterior.marginal_loglik,
                initial_probs=initial_probs,
                trans_probs=trans_probs,
                sum_w=sum_w,
                normd_x=normd_x,
                normd_xxT=normd_xxT
            )
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def m_step(self, batch_emissions, batch_posteriors, **kwargs):
        # Initial distribution
        initial_probs = jnp.sum(batch_posteriors.initial_probs, axis=0)
        self._initial_probs = tfd.Dirichlet(1.0001 + initial_probs).mode()

        # Transition distribution
        trans_probs = jnp.sum(batch_posteriors.trans_probs, axis=0)
        self._transition_matrix = tfd.Dirichlet(1.0001 + trans_probs).mode()

        # Gaussian emission distribution
        emission_dim = batch_posteriors.normd_x.shape[-1]
        normd_w = batch_posteriors.sum_w / jnp.sum(batch_posteriors.sum_w, axis=0)
        self._emission_means = jnp.sum(batch_posteriors.normd_x * normd_w[...,None], axis=0)
        self._emission_covs = (
            jnp.sum(batch_posteriors.normd_xxT * normd_w[..., None, None], axis=0)
            - jnp.einsum("ki,kj->kij", self._emission_means, self._emission_means)
            + 1e-4 * jnp.eye(emission_dim)
        )

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            tfb.SoftmaxCentered().inverse(self.initial_probabilities),
            tfb.SoftmaxCentered().inverse(self.transition_matrix),
            self.emission_means,
            PSDToRealBijector.forward(self.emission_covariance_matrices),
        )

    @unconstrained_params.setter
    def unconstrained_params(self, unconstrained_params):
        self._initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        self._transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        self._emission_means = unconstrained_params[2]
        self._emission_covs = PSDToRealBijector.inverse(unconstrained_params[3])
        