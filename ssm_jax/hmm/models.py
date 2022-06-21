from abc import ABC
from abc import abstractclassmethod
from abc import abstractproperty
from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.learning import compute_transition_probs
from ssm_jax.hmm.learning import hmm_fit_sgd
from ssm_jax.utils import PSDToRealBijector

# Using TFP for now since it has all our distributions
# (Distrax doesn't have Poisson, it seems.)


class BaseHMM(ABC):

    def __init__(self, initial_probabilities, transition_matrix):
        """Abstract base class for Hidden Markov Models.
        Child class specifies the emission distribution.

        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        num_states = transition_matrix.shape[-1]

        # Check shapes
        assert initial_probabilities.shape == (num_states,)
        assert transition_matrix.shape == (num_states, num_states)

        # Construct the  distribution objects
        self._initial_distribution = tfd.Categorical(probs=initial_probabilities)
        self._transition_distribution = tfd.Categorical(probs=transition_matrix)

    # Properties to get various attributes of the model from underyling distribution objects
    @property
    def num_states(self):
        return self._initial_distribution.probs.shape[-1]

    @property
    def num_obs(self):
        return self._emission_distribution.event_shape[0]

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
        raise NotImplementedError

    def sample(self, key, num_timesteps):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution[state].sample(seed=key1)  # defined in child
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
        """Compute the log joint probability of the states and observations
        """
        lp = self.initial_distribution.log_prob(states[0])
        lp += self.transition_distribution[states[:-1]].log_prob(states[1:]).sum()
        lp += self.emission_distribution[states].log_prob(emissions).sum(0)
        return lp

    def _conditional_logliks(self, emissions):
        # Input: emissions(T,) for scalar, or emissions(T,D) for vector
        # Add extra dimension to emissions for broadcasting over states.
        # Becomes emissions(T,:) or emissions(T,:,D) which broadcasts with emissions distribution
        # of shape (K,) or (K,D).
        log_likelihoods = self.emission_distribution.log_prob(emissions[:, None, ...])
        return log_likelihoods

    # Basic inference code
    def marginal_log_prob(self, emissions):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self.initial_probabilities, self.transition_matrix,
                                  self._conditional_logliks(emissions))

    def filter(self, emissions):
        """Compute filtering distribution."""
        return hmm_filter(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))

    def smoother(self, emissions):
        """Compute smoothing distribution."""
        return hmm_smoother(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """

        def _single_e_step(emissions):
            # TODO: do we need to use dynamic slice?

            posterior = hmm_smoother(self.initial_probabilities, self.transition_matrix,
                                     self._conditional_logliks(emissions))

            # Compute the transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)
            '''
            # Pad the posterior expectations to be the same length
            pad = jnp.zeros((len(emissions), self.num_states))
            padded_posterior = HMMPosterior(marginal_log_lkhd=posterior.marginal_loglik,
                                            filtered_probs=jnp.row_stack(posterior.filtered_probs, pad),
                                            predicted_probs=jnp.row_stack(posterior.predicted_probs, pad),
                                            smoothed_probs=jnp.row_stack(posterior.smoothed_probs, pad))
            '''
            return posterior, trans_probs

        return lax.map(_single_e_step, batch_emissions)

    # @classmethod
    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(1e-2), num_iters=50):
        """_summary_

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        hypers = self.hyperparams

        def _single_expected_log_joint(hmm, emissions, posterior, trans_probs):
            # TODO: do we need to use dynamic slice?
            log_likelihoods = vmap(hmm.emission_distribution.log_prob)(emissions)
            expected_states = posterior.smoothed_probs

            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * log_likelihoods)
            return lp

        def neg_expected_log_joint(params):
            hmm = self.from_unconstrained_params(params, hypers)
            f = vmap(partial(_single_expected_log_joint, hmm))
            lps = f(batch_emissions, batch_posteriors, batch_trans_probs)
            return -jnp.sum(lps / jnp.ones_like(batch_emissions).sum())

        # TODO: minimize the negative expected log joint with SGD
        hmm, losses = hmm_fit_sgd(self, batch_emissions, optimizer, num_iters, neg_expected_log_joint)
        return hmm, -losses

    # Properties to allow unconstrained optimization and JAX jitting
    @abstractproperty
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        raise NotImplementedError

    @abstractclassmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        raise NotImplementedError

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

    '''
    def m_step(self, emissions, posterior):
        raise NotImplementedError
    '''


@register_pytree_node_class
class BernoulliHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_probs):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_distribution = tfd.Independent(tfd.Bernoulli(probs=emission_probs), reinterpreted_batch_ndims=1)

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
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix), tfb.Sigmoid().inverse(self.emission_probs))

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.Sigmoid().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)


@register_pytree_node_class
class CategoricalHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_probs):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_distribution = tfd.Categorical(probs=emission_probs)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.dirichlet(key3, jnp.ones(emission_dim), (num_states,))
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
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix),
                tfb.SoftmaxCentered().inverse(self.emission_probs))

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.SoftmaxCentered().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)


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
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix), self.emission_means,
                PSDToRealBijector.forward(self.emission_covariance_matrices))

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
            posterior = hmm_smoother(self.initial_probabilities, self.transition_matrix,
                                     self._conditional_logliks(emissions))

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
        emission_covs = stats.sum_xxT / stats.sum_w[:, None, None] \
            - jnp.einsum('ki,kj->kij', emission_means, emission_means) \
            + 1e-4 * jnp.eye(emission_dim)

        # Pack the results into a new GaussianHMM
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)


@register_pytree_node_class
class PoissonHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_rates):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_distribution = tfd.Independent(tfd.Poisson(emission_rates), reinterpreted_batch_ndims=1)

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
        # TODO: does this work for Independent product of Poisson?
        return self._emission_distribution.distribution.rate

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix), tfb.Softplus().inverse(self.emission_rates))

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_rates = tfb.Softplus().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_rates, *hypers)
