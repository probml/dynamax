from abc import ABC
from abc import abstractclassmethod
from abc import abstractproperty
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax
from jax import vmap
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.hmm.learning import hmm_fit_sgd


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
        """Compute the log joint probability of the states and observations"""
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
        return hmm_posterior_mode(
            self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions)
        )

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

            posterior = hmm_two_filter_smoother(
                self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions)
            )

            # Compute the transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)
            return posterior, trans_probs

        return vmap(_single_e_step)(batch_emissions)

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
        """Helper property to get a PyTree of unconstrained parameters."""
        raise NotImplementedError

    @abstractclassmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        raise NotImplementedError

    @property
    def hyperparams(self):
        """Helper property to get a PyTree of model hyperparameters."""
        return tuple()

    # Use the to/from unconstrained properties to implement JAX tree_flatten/unflatten
    def tree_flatten(self):
        children = self.unconstrained_params
        aux_data = self.hyperparams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_unconstrained_params(children, aux_data)
