import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.tree_util import register_pytree_node_class

# Using TFP for now since it has all our distributions
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from ssm_jax.hmm.models.base import BaseHMM

import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from ssm_jax.utils import PSDToRealBijector


@register_pytree_node_class
class AutoregressiveHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 dynamics_matrices,
                 dynamics_biases,
                 dynamics_covariance_matrices):
        super(AutoregressiveHMM, self).__init__(initial_probabilities,
                                                transition_matrix)

        self._dynamics_matrices = dynamics_matrices
        self._dynamics_biases = dynamics_biases
        self._dynamics_covariance_matrices = dynamics_covariance_matrices

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim, num_lags=1):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        dynamics_matrices = jnp.zeros((num_states, num_lags, emission_dim, emission_dim))
        dynamics_biases = jr.normal(key3, (num_states, emission_dim))
        dynamics_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, dynamics_matrices, dynamics_biases, dynamics_covs)

    def sample(self, key, num_timesteps, initial_history):
        def _step(carry, key):
            state, history = carry
            key1, key2 = jr.split(key, 2)

            # Sample the next emission
            mean = jnp.einsum('lij,lj->i', self._dynamics_matrices[state], history)
            mean += self._dynamics_biases[state]
            cov = self._dynamics_covariance_matrices[state]
            emission = MVN(mean, cov).sample(seed=key1)

            next_state = self.transition_distribution[state].sample(seed=key2)
            next_history = jnp.row_stack([history[1:], emission])
            return (next_state, next_history), (state, emission, history)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution.sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions, history) = \
            lax.scan(_step, (initial_state, initial_history), keys)
        return states, emissions, history

    def emission_distribution(self, state, history):
        mean = jnp.einsum('lij,lj->i', self._dynamics_matrices[state], history)
        mean += self._dynamics_biases[state]
        cov = self._dynamics_covariance_matrices[state]
        return MVN(mean, cov)

    @property
    def emission_shape(self):
        return self._dynamics_biases.shape[1:]

    @property
    def dynamics_matrices(self):
        return self._dynamics_matrices

    @property
    def dynamics_biases(self):
        return self._dynamics_biases

    @property
    def dynamics_covariance_matrices(self):
        return self._dynamics_covariance_matrices

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            tfb.SoftmaxCentered().inverse(self.initial_probabilities),
            tfb.SoftmaxCentered().inverse(self.transition_matrix),
            self.dynamics_matrices,
            self.dynamics_biases,
            PSDToRealBijector.forward(self.dynamics_covariance_matrices),
        )

    @unconstrained_params.setter
    def unconstrained_params(self, unconstrained_params):
        self._initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        self._transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        self._dynamics_matrices = unconstrained_params[2]
        self._dynamics_biases = unconstrained_params[3]
        self._dynamics_covariance_matrices = PSDToRealBijector.inverse(unconstrained_params[4])

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        dynamics_matrices = unconstrained_params[2]
        dynamics_biases = unconstrained_params[3]
        dynamics_covariance_matrices = PSDToRealBijector.inverse(unconstrained_params[4])
        return cls(initial_probabilities,
                   transition_matrix,
                   dynamics_matrices,
                   dynamics_biases,
                   dynamics_covariance_matrices,
                   *hypers)

    # # Expectation-maximization (EM) code
    # def e_step(self, batch_emissions, batch_history):
    #     """The E-step computes expected sufficient statistics under the
    #     posterior. In the Gaussian case, this these are the first two
    #     moments of the data
    #     """

    #     @chex.dataclass
    #     class ARHMMSuffStats:
    #         # Wrapper for sufficient statistics of a GaussianHMM
    #         initial_probs: chex.Array
    #         sum_trans_probs: chex.Array
    #         sum_w: chex.Array
    #         sum_x: chex.Array       # emissions
    #         sum_u: chex.Array       # history
    #         sum_xxT: chex.Array     # emissions x emissions
    #         sum_uxT: chex.Array     # history x emissions
    #         sum_uuT: chex.Array     # history x history

    #     def _single_e_step(emissions, history):
    #         # Run the smoother
    #         posterior = hmm_smoother(
    #             self.initial_probabilities,
    #             self.transition_matrix,
    #             self._conditional_logliks(emissions, history)
    #         )

    #         # Compute the initial state and transition probabilities
    #         initial_probs = posterior.smoothed_probs[0]
    #         sum_trans_probs = compute_transition_probs(self.transition_matrix, posterior)

    #         # Compute the expected sufficient statistics
    #         sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
    #         sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)
    #         sum_u = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, history)
    #         sum_xxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, emissions, emissions)
    #         sum_uxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, history, emissions)
    #         sum_uuT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, history, history)

    #         # TODO: might need to normalize x_sum and xxT_sum for numerical stability
    #         stats = ARHMMSuffStats(
    #             initial_probs=initial_probs,
    #             sum_trans_probs=sum_trans_probs,
    #             sum_w=sum_w,
    #             sum_x=sum_x,
    #             sum_u=sum_u,
    #             sum_xxT=sum_xxT
    #             sum_uxT=sum_uxT
    #             sum_uuT=sum_uuT
    #         )
    #         return stats, posterior.marginal_loglik

    #     # Map the E step calculations over batches
    #     return vmap(_single_e_step)(batch_emissions, batch_history)

    # def m_step(self,
    #            batch_emissions,
    #            batch_posteriors,
    #            batch_history):

    #     # Sum the statistics across all batches
    #     stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

    #     # Initial distribution
    #     self._initial_probabilities = tfd.Dirichlet(1.0001 + stats.initial_probs).mode()

    #     # Transition distribution
    #     self._transition_matrix = tfd.Dirichlet(1.0001 + stats.sum_trans_probs).mode()

    #     # Gaussian emission distribution
    #     emission_dim = stats.sum_x.shape[-1]

    #     def _single_m_step(stats):
    #         # Compute MNIW posterior
    #         sum_u1u1T = jnp.block([[stats.sum_uuT,        stats.sum_u[:, None]],
    #                                [stats.sum_u[None, :], jnp.array([[stats.sum_w]])]])

    #         sum_u1xT = jnp.block([[stats.sum_uxT],
    #                               [stats.sum_x[None, :]]])

    #         weights_and_bias = jnp.linalg.solve(sum_u1u1T, sum_u1xT)
    #         # Return MNIW posterior mode



    #     self._emission_means = stats.sum_x / stats.sum_w[:, None]
    #     self._emission_covs = (
    #         stats.sum_xxT / stats.sum_w[:, None, None]
    #         - jnp.einsum("ki,kj->kij", self._emission_means, self._emission_means)
    #         + 1e-4 * jnp.eye(emission_dim)
    #     )

