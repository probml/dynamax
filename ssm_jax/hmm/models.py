import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.tree_util import register_pytree_node_class
import jax.scipy.special as jss
from jax.nn import sigmoid, softmax

from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty

from .core import hmm_filter, hmm_smoother

# TFP has the Poisson distribution but distrax doesn't?!
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

PSDToUnconstrained = tfb.Chain([
    # step 3: flatten the lower triangular portion of the matrix
    tfb.Invert(tfb.FillTriangular()),
    # step 2: take the log of the diagonals
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
    # step 1: decompose the precision matrix into its Cholesky factors
    tfb.Invert(tfb.CholeskyOuterProduct()),
])

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
        lp += self.emission_distribution[states].log_prob(emissions).sum()
        return lp

    def marginal_log_prob(self, emissions):
        log_likelihoods = self.emission_distribution.log_prob(emissions[...,None,:])
        return hmm_filter(self.initial_probabilities,
                            self.transition_matrix,
                            log_likelihoods)[0]

    # Properties to allow unconstrained optimization and JAX jitting
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
        # We have to be a little fancy since this classmethod
        # is inherited by subclasses with different constructors.
        return cls.from_unconstrained_params(children, aux_data)


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

        self._emissions_distribution = tfd.Independent(
            tfd.Bernoulli(probs=emission_probs),
            reinterpreted_batch_ndims=1)

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
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix)

        self._emission_distribution = tfd.Categorical(probs=emission_probs)

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
               PSDToUnconstrained.forward(self.emission_covariance_matrices)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_means = unconstrained_params[2]
        emission_covs = PSDToUnconstrained.inverse(unconstrained_params[3])
        return cls(initial_probabilities, transition_matrix, emission_means, emission_covs, *hypers)


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

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def emission_rates(self):
        return self.emission_distribution.rate

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