from abc import abstractmethod

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax import jit
from jax import lax
from jax import tree_map, tree_leaves
from functools import partial
from tqdm.auto import trange

from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.abstractions import SSM, Parameter
from ssm_jax.optimize import run_sgd

class BaseHMM(SSM):

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

    def initial_distribution(self):
        return tfd.Categorical(probs=self._initial_probs.value)

    def transition_distribution(self, state):
        return tfd.Categorical(probs=self._transition_matrix.value[state])

    @abstractmethod
    def emission_distribution(self, state):
        """Return a distribution over emissions given current state.

        Args:
            state (PyTree): current latent state.

        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

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
    @property
    def suff_stats_event_shape(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        raise NotImplementedError

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

    def fit_stochastic_em(self,
                          batch_emissions,
                          learn_rate_cool_down_frac=0.9,
                          batch_size=1,
                          num_epochs=50,
                          shuffle=True,
                          key=jr.PRNGKey(0),
        ):

        """
        Fit this HMM by running Stochastic Expectation-Maximization.

        Note that batch_emissions is initially of shape (N,T) where N is the
        number of independent sequences and T is the length of a sequence.
        Then a random subset of the sequences, and not of the timesteps, is
        sampledat each sub-epoch, with shape (B,T) where B is batch size.

        Note that this method assumes the instance's E-step returns sufficient
        statistics, instead of a more generic HMMPosterior object.

        This method also uses an exponentially decaying learning rate to anneal
        the minibatch sufficient statistics at each stage of training. It is
        recommended to set a value >0.5, and typically best to set a value >0.9.

        Args:
            batch_emissions (chex.Array): Independent sequences.
            learn_rate_cool_down_frac (float): Fraction of total training at
                which learning rate "cools down". Must be in range (0,1].
            batch_size (int): Number of sequences used at each update step.
            num_epochs (int): Iterations made through entire dataset.
            shuffle (bool): Indicates whether to shuffle minibatches.
            key (chex.PRNGKey): RNG key to shuffle minibatches.

        Returns:
            log_probs (chex.Array): Approximate log likelihood at each epoch
            TODO What's the most appropriate name for this value? It's not exactly
            the log likelihood, and not directly comparable to the log probs of
            the exact EM algorithm....
        """

        def _sample_minibatches(key, dataset, batch_size, shuffle):
            """Sequence generator."""
            n_data = len(tree_leaves(dataset)[0])
            perm = jnp.where(shuffle, jr.permutation(key, n_data), jnp.arange(n_data))
            for idx in range(0, n_data, batch_size):
                yield tree_map(lambda x: x[perm[idx:min(idx + batch_size, n_data)]], dataset)
        
        @jit
        def train_step(carry, input):
            self.unconstrained_params, rolling_stats = carry
            key, learning_rates = input

            sample_generator = \
                _sample_minibatches(key, batch_emissions, batch_size, shuffle)

            def _minibatch_step(carry, lrate):
                self.unconstrained_params, rolling_stats = carry

                emissions = next(sample_generator)
                batch_posterior_stats = self.e_step(emissions)
                
                these_stats = tree_map(
                    partial(jnp.sum, axis=0, keepdims=True),                    # keepdims=True important for consistency of shape for first minibatch
                    batch_posterior_stats
                )                
                
                rolling_stats = lax.cond(
                    jnp.all(rolling_stats.initial_probs==0),
                    partial(tree_map, lambda  _, s1:  lrate * scale * s1),
                    partial(tree_map, lambda s0, s1: (1-lrate) * s0 + lrate * scale * s1,),
                    rolling_stats, these_stats
                )
                
                self.m_step(emissions, rolling_stats)

                return (self.unconstrained_params, rolling_stats), these_stats.marginal_loglik.sum()

            # ------------------------------------------------------------------
            
            (params, rolling_stats), minibath_log_probs = lax.scan(
                _minibatch_step,
                (self.unconstrained_params, rolling_stats),
                learning_rates
            )

            return (params, rolling_stats), minibath_log_probs[-1]

        num_complete_batches, leftover = jnp.divmod(len(batch_emissions), batch_size)
        num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

        scale = len(batch_emissions) / batch_size

        # Initialize rolling sufficient statistics with 0-arrays
        init_stats = tree_map(
            lambda shp: jnp.zeros((1,) + shp),                                  # Add batch axis for M-step
            self.suff_stats_event_shape,
            is_leaf=lambda x: isinstance(x, tuple)                              # Tree leaves are the shape tuples
        )

        # Set learning rates
        total_iters = num_epochs * num_batches
        schedule = optax.exponential_decay(
            init_value=1.,
            end_value=0.,
            transition_steps=total_iters,
            decay_rate=total_iters**(-1./learn_rate_cool_down_frac),
        )
        learning_rates = schedule(jnp.arange(total_iters))

        (params, _), log_probs = lax.scan( \
            train_step,
            (self.unconstrained_params, init_stats),
            (jr.split(key, num_epochs), learning_rates.reshape(num_epochs, num_batches))
        )

        self.unconstrained_params = params

        return log_probs