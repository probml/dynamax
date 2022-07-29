from abc import abstractmethod
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax
from jax import vmap
from jax import jit
from jax import lax
from jax import vmap
from tqdm.auto import trange

from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.abstractions import Module, Parameter
from ssm_jax.optimize import run_sgd

class BaseHMM(Module):

    def __init__(self, initial_probabilities, transition_matrix):
        """
        Abstract base class for Hidden Markov Models.
        Child class specifies the emission distribution.

        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        # Check shapes
        num_states = transition_matrix.shape[-1]
        assert initial_probabilities.shape == (num_states,)
        assert transition_matrix.shape == (num_states, num_states)

        # Store the parameters
        self._initial_probs = Parameter(initial_probabilities, bijector=tfb.Invert(tfb.SoftmaxCentered()))
        self._transition_matrix = Parameter(transition_matrix, bijector=tfb.Invert(tfb.SoftmaxCentered()))

    # Properties to get various attributes of the model.
    @property
    def num_states(self):
        return self.initial_distribution().probs_parameter().shape[0]

    @property
    def num_obs(self):
        return self.emission_distribution(0).event_shape[0]

    @property
    def initial_probs(self):
        return self._initial_probs

    @property
    def transition_matrix(self):
        return self._transition_matrix

    # The following three functions define a state space model
    def initial_distribution(self):
        return tfd.Categorical(probs=self._initial_probs.value)

    def transition_distribution(self, state):
        return tfd.Categorical(probs=self._transition_matrix.value[state])

    @abstractmethod
    def emission_distribution(self, state):
        raise NotImplementedError

    def sample(self, key, num_timesteps):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution(state).sample(seed=key1)
            next_state = self.transition_distribution(state).sample(seed=key2)
            return next_state, (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)
        return states, emissions

    def log_prob(self, states, emissions):
        """Compute the log joint probability of the states and observations"""
        lp = self.initial_distribution().log_prob(states[0])
        lp += self.transition_distribution(states[:-1]).log_prob(states[1:]).sum()
        f = lambda state, emission: self.emission_distribution(state).log_prob(emission)
        lp += vmap(f)(states, emissions).sum()
        return lp

    def _conditional_logliks(self, emissions):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission: vmap(lambda state: self.emission_distribution(state).log_prob(emission))(jnp.arange(
            self.num_states))
        return vmap(f)(emissions)

    # Basic inference code
    def marginal_log_prob(self, emissions):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self.initial_probs.value,
                          self.transition_matrix.value,
                          self._conditional_logliks(emissions))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self.initial_probs.value,
                                  self.transition_matrix.value,
                                  self._conditional_logliks(emissions))

    def filter(self, emissions):
        """Compute filtering distribution."""
        return hmm_filter(self.initial_probs.value,
                          self.transition_matrix.value,
                          self._conditional_logliks(emissions))

    def smoother(self, emissions):
        """Compute smoothing distribution."""
        return hmm_smoother(self.initial_probs.value,
                            self.transition_matrix.value,
                            self._conditional_logliks(emissions))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """

        def _single_e_step(emissions):
            # TODO: do we need to use dynamic slice?
            posterior = hmm_two_filter_smoother(self.initial_probs.value,
                                                self.transition_matrix.value,
                                                self._conditional_logliks(emissions))

            # Compute the transition probabilities
            posterior.trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            return posterior

        return vmap(_single_e_step)(batch_emissions)

    def m_step(self, batch_emissions, batch_posteriors,
               optimizer=optax.adam(1e-2),
               num_mstep_iters=50):
        """_summary_

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        def neg_expected_log_joint(params, minibatch):
            minibatch_emissions, minibatch_posteriors = minibatch
            self.unconstrained_params = params

            def _single_expected_log_joint(emissions, posterior):
                log_likelihoods = self._conditional_logliks(emissions)
                expected_states = posterior.smoothed_probs
                trans_probs = posterior.trans_probs

                lp = jnp.sum(expected_states[0] * jnp.log(self.initial_probs.value))
                lp += jnp.sum(trans_probs * jnp.log(self.transition_matrix.value))
                lp += jnp.sum(expected_states * log_likelihoods)
                return lp

            lps = vmap(_single_expected_log_joint)(minibatch_emissions, minibatch_posteriors)
            return -jnp.sum(lps / batch_emissions.size)

        # Minimize the negative expected log joint with SGD
        params, losses = run_sgd(neg_expected_log_joint,
                                 self.unconstrained_params,
                                 (batch_emissions, batch_posteriors),
                                 optimizer=optimizer,
                                 num_epochs=num_mstep_iters)
        self.unconstrained_params = params

    def fit_em(self, batch_emissions, num_iters=50, **kwargs):
        """Fit this HMM with Expectation-Maximization (EM).

        Args:
            batch_emissions (_type_): _description_
            num_iters (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        @jit
        def em_step(params):
            self.unconstrained_params = params
            batch_posteriors = self.e_step(batch_emissions)
            self.m_step(batch_emissions, batch_posteriors, **kwargs)
            return self.unconstrained_params, batch_posteriors

        log_probs = []
        params = self.unconstrained_params
        for _ in trange(num_iters):
            params, batch_posteriors = em_step(params)
            log_probs.append(batch_posteriors.marginal_loglik.sum())

        self.unconstrained_params = params
        return jnp.array(log_probs)

    def fit_sgd(self,
                batch_emissions,
                optimizer=optax.adam(1e-3),
                batch_size=1,
                num_epochs=50,
                shuffle=False,
                key=jr.PRNGKey(0),
        ):
        """
        Fit this HMM by running SGD on the marginal log likelihood.

        Note that batch_emissions is initially of shape (N,T)
        where N is the number of independent sequences and
        T is the length of a sequence. Then, a random susbet with shape (B, T)
        of entire sequence, not time steps, is sampled at each step where B is
        batch size.

        Args:
            batch_emissions (chex.Array): Independent sequences.
            optmizer (optax.Optimizer): Optimizer.
            batch_size (int): Number of sequences used at each update step.
            num_epochs (int): Iterations made through entire dataset.
            shuffle (bool): Indicates whether to shuffle minibatches.
            key (chex.PRNGKey): RNG key to shuffle minibatches.

        Returns:
            losses: Output of loss_fn stored at each step.
        """
        def _loss_fn(params, minibatch_emissions):
            """Default objective function."""
            self.unconstrained_params = params
            f = lambda emissions: -self.marginal_log_prob(emissions) / len(emissions)
            return vmap(f)(minibatch_emissions).mean()

        params, losses = run_sgd(_loss_fn,
                                 self.unconstrained_params,
                                 batch_emissions,
                                 optimizer=optimizer,
                                 batch_size=batch_size,
                                 num_epochs=num_epochs,
                                 shuffle=shuffle,
                                 key=key)
        self.unconstrained_params = params
        return losses
