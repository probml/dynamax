from abc import ABC, abstractmethod
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
from ssm_jax.utils import sgd_helper


class BaseHMM(ABC):
    def __init__(self, initial_probabilities, transition_matrix):
        """
        Abstract base class for Hidden Markov Models.
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
        self._initial_probabilities = initial_probabilities
        self._transition_matrix = transition_matrix

    # Properties to get various attributes of the model from underyling distribution objects
    @property
    def num_states(self):
        return len(self._initial_probabilities)

    @property
    @abstractmethod
    def emission_shape(self):
        raise NotImplementedError

    @property
    def initial_probabilities(self):
        return self._initial_probabilities

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def initial_distribution(self):
        return tfd.Categorical(probs=self.initial_probabilities)

    @property
    def transition_distribution(self):
        return tfd.Categorical(probs=self.transition_matrix)

    @abstractmethod
    def emission_distribution(self, state, **covariates):
        raise NotImplementedError

    def sample(self, key, num_timesteps, **covariates):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(state, key, **covariate):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution(state, **covariate).sample(seed=key1)
            next_state = self.transition_distribution[state].sample(seed=key2)
            return next_state, (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution.sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)
        return states, emissions

    def log_prob(self, states, emissions, **covariates):
        """Compute the log joint probability of the states and observations"""
        lp = self.initial_distribution.log_prob(states[0])
        lp += self.transition_distribution[states[:-1]].log_prob(states[1:]).sum()
        f = lambda state, emission, **covariate: \
            self.emission_distribution(state, **covariate).log_prob(emission)
        lp += vmap(f)(states, emissions, **covariates).sum()
        return lp

    def _conditional_logliks(self, emissions, **covariates):
        # Compute the log probability for each time step.
        # NOTE: This assumes each covariate is a time series
        #       of the same length as the emissions. We could consider having another
        #       argument for `metadata` that is static.

        # Perform a nested vmap over timeteps and states
        f = lambda emission, **covariate: \
            vmap(lambda state: \
                self.emission_distribution(state, **covariate).log_prob(emission))(
                    jnp.arange(self.num_states)
                )
        return vmap(f)(emissions, **covariates)

    # Basic inference code
    def marginal_log_prob(self, emissions, **covariates):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self.initial_probabilities,
                          self.transition_matrix,
                          self._conditional_logliks(emissions, **covariates))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions, **covariates):
        """Compute Viterbi path."""
        return hmm_posterior_mode(
            self.initial_probabilities,
            self.transition_matrix,
            self._conditional_logliks(emissions, **covariates)
        )

    def filter(self, emissions, **covariates):
        """Compute filtering distribution."""
        return hmm_filter(self.initial_probabilities,
                          self.transition_matrix,
                          self._conditional_logliks(emissions, **covariates))

    def smoother(self, emissions, **covariates):
        """Compute smoothing distribution."""
        return hmm_smoother(self.initial_probabilities,
                            self.transition_matrix,
                            self._conditional_logliks(emissions, **covariates))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions, **batch_covariates):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """

        def _single_e_step(emissions, **covariates):
            # TODO: do we need to use dynamic slice?

            posterior = hmm_two_filter_smoother(
                self.initial_probabilities,
                self.transition_matrix,
                self._conditional_logliks(emissions, **covariates)
            )

            # Compute the transition probabilities
            trans_probs = compute_transition_probs(
                self.transition_matrix, posterior)

            return (posterior, trans_probs), posterior.marginal_loglik

        return vmap(_single_e_step)(batch_emissions, **batch_covariates)

    def m_step(self,
               batch_emissions,
               batch_posteriors,
               optimizer=optax.adam(1e-2),
               num_iters=50,
               **batch_covariates):
        """_summary_

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        hypers = self.hyperparams
        cls = self.__class__

        def _single_expected_log_joint(params, emissions, posterior, **covariates):
            # TODO: Handle variable length time series
            hmm = cls.from_unconstrained_params(params, hypers)
            state_posterior, trans_probs = posterior
            expected_states = state_posterior.smoothed_probs
            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * hmm._conditional_logliks(emissions, **covariates))
            return lp

        def neg_expected_log_joint(params,
                                   minibatch_emissions,
                                   minibatch_posteriors,
                                   **minibatch_covariates):
            f = vmap(partial(_single_expected_log_joint, params))
            lps = f(minibatch_emissions, minibatch_posteriors, **minibatch_covariates)
            return -jnp.sum(lps) / minibatch_emissions.size

        params, losses = sgd_helper(neg_expected_log_joint,
                                    self.unconstrained_params,
                                    batch_emissions,
                                    batch_args=(batch_posteriors,),
                                    batch_kwargs=batch_covariates,
                                    optimizer=optimizer,
                                    num_iters=num_iters)

        self.unconstrained_params = params

    # Properties to allow unconstrained optimization and JAX jitting
    @property
    @abstractmethod
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        raise NotImplementedError

    @unconstrained_params.setter
    @abstractmethod
    def unconstrained_params(self, value):
        raise NotImplementedError

    @classmethod
    @abstractmethod
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
