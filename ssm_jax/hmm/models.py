from abc import ABC, abstractclassmethod, abstractproperty
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.tree_util import register_pytree_node_class, tree_map

import chex

# Using TFP for now since it has all our distributions
# (Distrax doesn't have Poisson, it seems.)
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from ssm_jax.hmm.inference import (
    HMMPosterior,
    hmm_filter,
    hmm_smoother,
    hmm_posterior_mode,
    compute_transition_probs)
from ssm_jax.utils import PSDToRealBijector


class BaseHMM(ABC):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix):
        """Abstract base class for Hidden Markov Models.

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            initial_probs_concentration (float, optional): _description_.
                Defaults to 1.0001.
            transition_matrix_concentration (float, optional): _description_.
                Defaults to 1.0001.
        """
        num_states = transition_matrix.shape[-1]

        # Check shapes
        assert initial_probabilities.shape == (num_states,)
        assert transition_matrix.shape == (num_states, num_states)

        # Construct the model from distrax distributions
        self._initial_distribution = tfd.Categorical(probs=initial_probabilities)
        self._transition_distribution = tfd.Categorical(probs=transition_matrix)

    # Properties to get various attributes of the model
    @property
    def num_states(self):
        return self._initial_distribution.probs.shape[-1]

    @property
    def emission_shape(self):
        return self.emission_distribution.event_shape

    @property
    def num_obs(self):
        return self.emission_distribution.event_shape[0]

    @property
    def initial_probabilities(self):
        return self._initial_distribution.probs

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs

    @property
    def initial_distribution(self):
        return self._initial_distribution

    @property
    def transition_distribution(self):
        return self._transition_distribution

    @abstractproperty
    def emission_distribution(self):
        raise NotImplemented

    def sample(self, key, num_timesteps):
        """Sample a sequence of latent states and emissions.

        Args:
            key (_type_): _description_
            num_timesteps (_type_): _description_
        """
        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution[state].sample(seed=key1)
            next_state = self.transition_distribution[state].sample(seed=key2)
            return next_state, (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution.sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)
        return states, emissions

    def log_prob(self, states, emissions):
        """Compute the log probability of the states and data
        """
        lp = self.initial_distribution.log_prob(states[0])
        lp += self.transition_distribution[states[:-1]].log_prob(states[1:]).sum()
        lp += self.emission_distribution[states].log_prob(emissions).sum(0)
        return lp

    ### Basic inference code
    def marginal_log_prob(self, emissions):
        # Add extra dimension to emissions for broadcasting over states.
        log_likelihoods = self.emission_distribution.log_prob(emissions[:,None,...])
        return hmm_filter(self.initial_probabilities,
                            self.transition_matrix,
                            log_likelihoods)[0]

    def most_likely_states(self, emissions):
        # Add extra dimension to emissions for broadcasting over states.
        log_likelihoods = self.emission_distribution.log_prob(emissions[:,None,...])
        return hmm_posterior_mode(self.initial_probabilities,
                                  self.transition_matrix,
                                  log_likelihoods)

    def filter(self, emissions):
        # Add extra dimension to emissions for broadcasting over states.
        log_likelihoods = self.emission_distribution.log_prob(emissions[:,None,...])
        return hmm_filter(self.initial_probabilities,
                            self.transition_matrix,
                            log_likelihoods)

    def smoother(self, emissions):
        # Add extra dimension to emissions for broadcasting over states.
        log_likelihoods = self.emission_distribution.log_prob(emissions[:,None,...])
        return hmm_smoother(self.initial_probabilities,
                            self.transition_matrix,
                            log_likelihoods)

    ### Expectation-maximization (EM) code
    def e_step(self, batch_emissions, batch_num_timesteps):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        def _single_e_step(emissions, num_timesteps):
            # TODO: do we need to use dynamic slice?
            emissions = lax.dynamic_slice_in_dim(emissions, 0, num_timesteps)

            posterior = hmm_smoother(self.initial_probabilities,
                                self.transition_matrix,
                                self.emission_distribution.log_prob(emissions[:, None, ...]))

            # Compute the transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Pad the posterior expectations to be the same length
            pad = jnp.zeros((len(emissions) - num_timesteps, self.num_states))
            padded_posterior = HMMPosterior(
                marginal_log_lkhd=posterior.marginal_log_lkhd,
                filtered_probs=jnp.row_stack(posterior.filtered_probs, pad),
                predicted_probs=jnp.row_stack(posterior.predicted_probs, pad),
                smoothed_probs=jnp.row_stack(posterior.smoothed_probs, pad))

            return padded_posterior, trans_probs

        return lax.map(_single_e_step, (batch_emissions, batch_num_timesteps))

    def m_step(self, batch_emissions, batch_num_timesteps, batch_posteriors, batch_trans_probs):
        """_summary_

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        def _single_expected_log_joint(hmm, emissions, num_timesteps, posterior, trans_probs):
            # TODO: do we need to use dynamic slice?
            log_likelihoods = hmm.emission_distribution.log_prob(
                emissions[:num_timesteps, None,...])
            expected_states = posterior.smoothed_probs[:num_timesteps]

            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * log_likelihoods)
            return lp

        def neg_expected_log_joint(hmm):
            f = vmap(partial(_single_expected_log_joint, hmm))
            lps = f(batch_emissions, batch_num_timesteps, batch_posteriors, batch_trans_probs)
            return -lps / jnp.sum(batch_num_timesteps)

        # TODO: minimize the negative expected log joint with SGD
        raise NotImplementedError()

    ### Properties to allow unconstrained optimization and JAX jitting
    @abstractproperty
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        raise NotImplemented

    @abstractclassmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        raise NotImplemented

    @property
    def hyperparams(self):
        """Helper property to get a PyTree of model hyperparameters.
        """
        return tuple()

    # Use the to/from unconstrained properties to implement JAX tree_flatten/unflatten
    def tree_flatten(self):
        children = self.unconstrained_params
        aux_data = self.hyperparams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_unconstrained_params(children, aux_data)

    def m_step(self, emissions, posterior):
        raise NotImplemented


@register_pytree_node_class
class BernoulliHMM(BaseHMM):
    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_probs):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix)

        self._emission_distribution = tfd.Independent(
            tfd.Bernoulli(probs=emission_probs),
            reinterpreted_batch_ndims=1)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.uniform(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_probs(self):
        return self.emission_distribution.distribution.probs_parameter()

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return tfb.SoftmaxCentered().inverse(self.initial_probabilities), \
               tfb.SoftmaxCentered().inverse(self.transition_matrix), \
               tfb.Sigmoid().inverse(self.emission_probs)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.Sigmoid().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)


@register_pytree_node_class
class CategoricalHMM(BaseHMM):
    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_probs):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix)

        self._emission_distribution = tfd.Categorical(probs=emission_probs)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.dirichlet(key1, jnp.ones(emission_dim), (num_states,))
        return cls(initial_probs, transition_matrix, emission_probs)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_probs(self):
        return self.emission_distribution.probs_parameter()

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return tfb.SoftmaxCentered().inverse(self.initial_probabilities), \
               tfb.SoftmaxCentered().inverse(self.transition_matrix), \
               tfb.SoftmaxCentered().inverse(self.emission_probs)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.SoftmaxCentered().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)


@register_pytree_node_class
class GaussianHMM(BaseHMM):
    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrices):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix)

        self._emission_distribution = tfd.MultivariateNormalFullCovariance(
            emission_means, emission_covariance_matrices)

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
        return self.emission_distribution.mean()

    @property
    def emission_covariance_matrices(self):
        return self.emission_distribution.covariance()

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return tfb.SoftmaxCentered().inverse(self.initial_probabilities), \
               tfb.SoftmaxCentered().inverse(self.transition_matrix), \
               self.emission_means, \
               PSDToRealBijector.forward(self.emission_covariance_matrices)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_means = unconstrained_params[2]
        emission_covs = PSDToRealBijector.inverse(unconstrained_params[3])
        return cls(initial_probabilities, transition_matrix, emission_means, emission_covs, *hypers)

    ### Expectation-maximization (EM) code
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
            posterior = hmm_smoother(self.initial_probabilities,
                                     self.transition_matrix,
                                     self.emission_distribution.log_prob(emissions[:, None, ...]))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            sum_trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum('tk->k', posterior.smoothed_probs)
            sum_x = jnp.einsum('tk, ti->ki', posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum('tk, ti, tj->kij', posterior.smoothed_probs, emissions, emissions)

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = GaussianHMMSuffStats(initial_probs=initial_probs,
                                         sum_trans_probs=sum_trans_probs,
                                         sum_w=sum_w,
                                         sum_x=sum_x,
                                         sum_xxT=sum_xxT)
            return stats, posterior.marginal_log_lkhd

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def m_step(self, batch_stats):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)

        # Initial distribution
        initial_probs = tfd.Dirichlet(1.0001 + stats.initial_probs).mode()

        # Transition distribution
        transition_matrix = tfd.Dirichlet(1.0001 + stats.sum_trans_probs).mode()

        # Gaussian emission distribution
        emission_dim = stats.sum_x.shape[-1]
        emission_means = stats.sum_x / stats.sum_w[:, None]
        emission_covs = stats.sum_xxT / stats.sum_w[:, None, None] \
            - jnp.einsum('ki,kj->kij', emission_means, emission_means) \
            + 1e-4 * jnp.eye(emission_dim)

        # Pack the results into a new GaussianHMM
        return GaussianHMM(initial_probs,
                           transition_matrix,
                           emission_means,
                           emission_covs)


@register_pytree_node_class
class PoissonHMM(BaseHMM):
    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_rates):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix)

        self._emission_distribution = tfd.Independent(
            tfd.Poisson(emission_rates), reinterpreted_batch_ndims=1)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_rates = jr.exponential(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_rates)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_rates(self):
        return self.emission_distribution.distribution.rate

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return tfb.SoftmaxCentered().inverse(self.initial_probabilities), \
               tfb.SoftmaxCentered().inverse(self.transition_matrix), \
               tfb.Softplus().inverse(self.emission_rates)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_rates = tfb.Softplus().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_rates, *hypers)
