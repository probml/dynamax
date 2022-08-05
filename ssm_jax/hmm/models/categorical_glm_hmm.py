from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.nn import one_hot
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class CategoricalRegressionHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_matrices,
                 emission_biases,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        # Check shapes
        # assert emission_probs.ndim == 3, \
        #     "emission_probs must be (num_states x num_classes)"
        self._num_classes = emission_biases.shape[0]
        # Save parameters and hyperparameters
        self._emission_matrices = Parameter(emission_matrices)
        self._emission_biases = Parameter(emission_biases)
        # self._emission_probs = Parameter(emission_probs, bijector=tfb.Invert(tfb.SoftmaxCentered()))
        self._emission_prior_concentration = Parameter(emission_prior_concentration  * jnp.ones(self._num_classes),
                                                       is_frozen=True,
                                                       bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, num_classes, feature_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_matrices = jr.normal(key3, (num_states, num_classes, feature_dim))
        emission_biases = jr.normal(key4, (num_states, num_classes))
        return cls(initial_probs, transition_matrix, emission_matrices, emission_biases)

    # Properties to get various parameters of the model
    @property
    def emission_matrices(self):
        return self._emission_matrices

    @property
    def emission_biases(self):
        return self._emission_biases

    @property
    def num_classes(self):
        return self.emission_probs.value.shape[2]

    # def emission_distribution(self, state, features=None):
    #     logits = self._emission_matrices.value[state] @ features + self._emission_biases.value[state]
    #     return tfd.Categorical(logits=logits)

    def emission_distribution(self, state, **covariates):
        logits = self._emission_matrices.value[state] @ covariates['features'] + self._emission_biases.value[state]
        return tfd.Categorical(logits=logits)

    # def _compute_conditional_logliks(self, emissions, **covariates):
    #     # Compute the log probability for each time step by
    #     # performing a nested vmap over emission time steps and states.
    #     f = lambda emission, **covariate: \
    #         vmap(lambda state: self.emission_distribution(state, **covariate).log_prob(emission))(
    #             jnp.arange(self.num_states))
    #     return vmap(f)(emissions, **covariates)

    # def log_prior(self):
    #     lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
    #     lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
    #     lp += tfd.Dirichlet(self._emission_prior_concentration.value).log_prob(self.emission_probs.value).sum()
    #     return lp

    # # Expectation-maximization (EM) code
    # def e_step(self, batch_emissions, batch_features=None):
    #     """The E-step computes expected sufficient statistics under the
    #     posterior. In the generic case, we simply return the posterior itself.
    #     """
    #     def _single_e_step(emissions, features):
    #         transition_matrices = self._compute_transition_matrices()
    #         posterior = hmm_two_filter_smoother(self._compute_initial_probs(),
    #                                             transition_matrices,
    #                                             self._compute_conditional_logliks(emissions, features=features))

    #         # Compute the transition probabilities
    #         posterior.trans_probs = compute_transition_probs(
    #             transition_matrices, posterior,
    #             reduce_sum=(transition_matrices.ndim == 2))

    #         return posterior

    #     return vmap(_single_e_step)(batch_emissions, batch_features)

    # Expectation-maximization (EM) code
    # def e_step(self, batch_emissions, **batch_covariates):
    #     """The E-step computes expected sufficient statistics under the
    #     posterior. In the generic case, we simply return the posterior itself.
    #     """
    #     def _single_e_step(emissions, **covariates):
    #         transition_matrices = self._compute_transition_matrices()
    #         posterior = hmm_two_filter_smoother(self._compute_initial_probs(),
    #                                             transition_matrices,
    #                                             self._compute_conditional_logliks(emissions, **covariates))

    #         # Compute the transition probabilities
    #         posterior.trans_probs = compute_transition_probs(
    #             transition_matrices, posterior,
    #             reduce_sum=(transition_matrices.ndim == 2))

    #         return posterior

    #     return vmap(_single_e_step)(batch_emissions, **batch_covariates)

    # def _m_step_emissions(self, batch_emissions,
    #                       batch_posteriors,
    #                       optimizer=optax.adam(1e-2),
    #                       num_mstep_iters=50,
    #                       **batch_covariates):

    #     def neg_expected_log_joint(params, minibatch):
    #         minibatch_emissions, minibatch_posteriors, minibatch_covariates = minibatch
    #         scale = len(batch_emissions) / len(minibatch_emissions)
    #         self.unconstrained_params = params

    #         def _single_expected_log_like(emissions, posterior, **covariates):
    #             log_likelihoods = self._conditional_logliks(emissions, **covariates)
    #             expected_states = posterior.smoothed_probs
    #             lp += jnp.sum(expected_states * log_likelihoods)
    #             return lp

    #         log_prior = self.log_prior()
    #         minibatch_ells = vmap(_single_expected_log_like)(
    #             minibatch_emissions, minibatch_posteriors, minibatch_covariates)
    #         expected_log_joint = log_prior + minibatch_ells.sum() * scale
    #         return -expected_log_joint / batch_emissions.size

    #     # Minimize the negative expected log joint with SGD
    #     params, losses = run_sgd(neg_expected_log_joint,
    #                              self.unconstrained_params,
    #                              (batch_emissions, batch_posteriors, batch_covariates),
    #                              optimizer=optimizer,
    #                              num_epochs=num_mstep_iters)
    #     self.unconstrained_params = params
