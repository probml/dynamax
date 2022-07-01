import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

# Using TFP for now since it has all our distributions
import tensorflow_probability.substrates.jax.distributions as tfd
MVN = tfd.MultivariateNormalFullCovariance

from ssm_jax.hmm.models import BaseHMM
from ssm_jax.hmm.inference import (
    HMMPosterior,
    hmm_filter,
    hmm_smoother,
    hmm_posterior_mode,
    compute_transition_probs)

class LinearRegression(object):
    pass

class AutoregressiveHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 dynamics_matrices,
                 dynamics_biases,
                 dynamics_covariance_matrices):
        super(AutoregressiveHMM, self).__init__(initial_probabilities,
                                                transition_matrix)

        self.dynamics_matrices = dynamics_matrices
        self.dynamics_biases = dynamics_biases
        self.dynamics_covariance_matrices = dynamics_covariance_matrices

    def sample(self, key, num_timesteps, history):
        def _step(carry, key):
            state, history = carry
            key1, key2 = jr.split(key, 2)

            # Sample the next emission
            mean = jnp.einsum('lij,lj->i', self.dynamics_matrices[state], history)
            mean += self.dynamics_biases[state]
            cov = self.dynamics_covariance_matrices[state]
            emission = MVN(mean, cov).sample(seed=key1)

            next_state = self.transition_distribution[state].sample(seed=key2)
            next_history = jnp.row_stack([history[1:], emission])
            return (next_state, next_history), (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution.sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, (initial_state, history), keys)
        return states, emissions


    def log_prob(self, states, emissions, history):
        lp = self.initial_distribution.log_prob(states[0])
        lp += self.transition_distribution[states[:-1]].log_prob(states[1:]).sum()

        def _compute_lp(history, args):
            state, emission = args
            mean = jnp.einsum('klij,lj->ki', self.dynamics_matrices[state], history)
            mean += self.dynamics_biases[state]
            cov += self.dynamics_covariance_matrices[state]
            lp = MVN(mean, )
            next_history = jnp.row_stack([history[1:], emission])
            return next_history, mean

        lp += self.emission_distribution[states].log_prob(emissions).sum(0)
        return lp

    def _conditional_logliks(self, emissions, history):

        def _compute_mean(history, emission):
            mean = jnp.einsum('klij,lj->ki', self.dynamics_matrices, history)
            mean += self.dynamics_biases
            next_history = jnp.row_stack([history[1:], emission])
            return next_history, mean

        means = lax.scan(_compute_mean, history, emissions)
        covs = self.dynamics_covariance_matrices
        return MVN(means, covs).log_prob(emissions[:, None, ...])

    def marginal_log_prob(self, emissions, history):
        posterior = hmm_filter(self.initial_probabilities, self.transition_matrix,
                               self._conditional_logliks(emissions))
        ll = posterior.marginal_loglik
        return ll

    def most_likely_states(self, emissions, history):
        return hmm_posterior_mode(self.initial_probabilities,
                                  self.transition_matrix,
                                  self._conditional_logliks(emissions, history))

    def filter(self, emissions, history):
        return hmm_filter(self.initial_probabilities,
                          self.transition_matrix,
                          self._conditional_logliks(emissions, history))

    def smoother(self, emissions, history):
        return hmm_smoother(self.initial_probabilities,
                            self.transition_matrix,
                            self._conditional_logliks(emissions, history))
