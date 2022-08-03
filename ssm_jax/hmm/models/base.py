from abc import abstractmethod

import jax.numpy as jnp
import jax.random as jr
from numpy import roll
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
from ssm_jax.optimize import run_sgd, sample_minibatches


class BaseHMM(SSM):

    # Properties to get various attributes of the model.
    # TODO: These need to be updated when we have covariates.
    @property
    def num_states(self):
        return self.initial_distribution().probs_parameter().shape[0]

    @property
    def num_obs(self):
        return self.emission_distribution(0).event_shape[0]

    # Three helper functions to compute the initial probabilities,
    # transition matrix (or matrices), and conditional log likelihoods.
    # These are the args to the HMM inference functions, and they can
    # be computed using the generic SSM initial_distribution(),
    # transition_distribution(), and emission_distribution() functions.
    def _compute_initial_probs(self):
        return self.initial_distribution().probs_parameter()

    def _compute_transition_matrices(self, **covariates):
        if len(covariates) > 0:
            f = lambda **covariate: \
                vmap(lambda state: \
                    self.transition_distribution(state, **covariate).probs_parameter())(
                        jnp.arange(self.num_states))
            return vmap(f)(**covariates)
        else:
            g = vmap(lambda state: self.transition_distribution(state).probs_parameter())
            return g(jnp.arange(self.num_states))

    def _compute_conditional_logliks(self, emissions):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission: \
            vmap(lambda state: self.emission_distribution(state).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions)

    # Basic inference code
    def marginal_log_prob(self, emissions):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self._compute_initial_probs(),
                          self._compute_transition_matrices(),
                          self._compute_conditional_logliks(emissions))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self._compute_initial_probs(),
                                  self._compute_transition_matrices(),
                                  self._compute_conditional_logliks(emissions))

    def filter(self, emissions):
        """Compute filtering distribution."""
        return hmm_filter(self._compute_initial_probs(),
                          self._compute_transition_matrices(),
                          self._compute_conditional_logliks(emissions))

    def smoother(self, emissions):
        """Compute smoothing distribution."""
        return hmm_smoother(self._compute_initial_probs(),
                            self._compute_transition_matrices(),
                            self._compute_conditional_logliks(emissions))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        # TODO: Assert batch_emissions is really batched!
        def _single_e_step(emissions):
            transition_matrices = self._compute_transition_matrices()
            posterior = hmm_two_filter_smoother(self._compute_initial_probs(),
                                                transition_matrices,
                                                self._compute_conditional_logliks(emissions))

            # Compute the transition probabilities
            posterior.trans_probs = compute_transition_probs(
                transition_matrices, posterior,
                reduce_sum=(transition_matrices.ndim == 2))

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
            scale = len(batch_emissions) / len(minibatch_emissions)
            self.unconstrained_params = params

            def _single_expected_log_joint(emissions, posterior):
                initial_probs = self._compute_initial_probs()
                trans_matrices = self._compute_transition_matrices()
                log_likelihoods = self._compute_conditional_logliks(emissions)
                expected_states = posterior.smoothed_probs
                trans_probs = posterior.trans_probs

                lp = jnp.sum(expected_states[0] * jnp.log(initial_probs))
                lp += jnp.sum(trans_probs * jnp.log(trans_matrices))
                lp += jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior()
            minibatch_lps = vmap(_single_expected_log_joint)(
                minibatch_emissions, minibatch_posteriors)
            expected_log_joint = log_prior + minibatch_lps.sum() * scale
            return -expected_log_joint / batch_emissions.size

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
            lp = self.log_prior() + batch_posteriors.marginal_loglik.sum()
            self.m_step(batch_emissions, batch_posteriors, **kwargs)
            return self.unconstrained_params, lp

        log_probs = []
        params = self.unconstrained_params
        for _ in trange(num_iters):
            params, lp = em_step(params)
            log_probs.append(lp)

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
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(self.marginal_log_prob)(minibatch_emissions)
            lp = self.log_prior() + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

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

class StandardHMM(BaseHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
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

        # And the hyperparameters of the prior
        self._initial_probs_concentration = Parameter(initial_probs_concentration * jnp.ones(num_states),
                                                      is_frozen=True,
                                                      bijector=tfb.Invert(tfb.Softplus()))
        self._transition_matrix_concentration = Parameter(transition_matrix_concentration * jnp.ones(num_states),
                                                          is_frozen=True,
                                                          bijector=tfb.Invert(tfb.Softplus()))

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

    def _m_step_initial_probs(self, batch_emissions, batch_posteriors):
        post = tfd.Dirichlet(self._initial_probs_concentration.value +
                             batch_posteriors.initial_probs.sum(axis=0))
        self._initial_probs.value = post.mode()

    def _m_step_transition_matrix(self, batch_emissions, batch_posteriors):
        post = tfd.Dirichlet(self._transition_matrix_concentration.value +
                             batch_posteriors.trans_probs.sum(axis=0))
        self._transition_matrix.value = post.mode()

    def _m_step_emissions(self, batch_emissions, batch_posteriors,
                          optimizer=optax.adam(1e-2),
                          num_mstep_iters=50):

        def neg_expected_log_joint(params, minibatch):
            minibatch_emissions, minibatch_posteriors = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            self.unconstrained_params = params

            def _single_expected_log_like(emissions, posterior):
                log_likelihoods = self._conditional_logliks(emissions)
                expected_states = posterior.smoothed_probs
                lp += jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior()
            minibatch_ells = vmap(_single_expected_log_like)(
                minibatch_emissions, minibatch_posteriors)
            expected_log_joint = log_prior + minibatch_ells.sum() * scale
            return -expected_log_joint / batch_emissions.size

        # Minimize the negative expected log joint with SGD
        params, losses = run_sgd(neg_expected_log_joint,
                                 self.unconstrained_params,
                                 (batch_emissions, batch_posteriors),
                                 optimizer=optimizer,
                                 num_epochs=num_mstep_iters)
        self.unconstrained_params = params

    def m_step(self, batch_emissions, batch_posteriors,
               optimizer=optax.adam(1e-2),
               num_mstep_iters=50):

        self._m_step_initial_probs(batch_emissions, batch_posteriors)
        self._m_step_transition_matrix(batch_emissions, batch_posteriors)
        self._m_step_emissions(batch_emissions, batch_posteriors,
                               optimizer=optimizer,
                               num_mstep_iters=num_mstep_iters)



class ExponentialFamilyHMM(StandardHMM):
    """TODO
    """

    @abstractmethod
    def _zeros_like_suff_stats(self):
        raise NotImplementedError


    def fit_stochastic_em(self,
                          batch_emissions,
                          schedule=None,
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
            schedule (optax schedule, Callable: int -> [0, 1]): learning rate schedule;
                Defaults to exponential schedule.
            batch_size (int): Number of sequences used at each update step.
            num_epochs (int): Iterations made through entire dataset.
            shuffle (bool): Indicates whether to shuffle minibatches.
            key (chex.PRNGKey): RNG key to shuffle minibatches.

        Returns:
            log_probs (chex.Array): Approximate log joint prob at each epoch

        TODOs:
            - allow for larger minibatches
            - implement shuffle
            - may need a while loop?
        """

        # Set up the minibatch stuff
        assert batch_size == 1, "TODO: Support larger minibatches"
        num_complete_batches, leftover = jnp.divmod(len(batch_emissions), batch_size)
        num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
        assert num_batches == len(batch_emissions)

        # Set the scaling factor for the minibatch statistics.
        # TODO: revisit this when we allow variable length time series
        scale = len(batch_emissions) / batch_size

        # Set learning rates
        if schedule is None:
            schedule = optax.exponential_decay(
                init_value=1.,
                end_value=0.,
                transition_steps=num_batches,
                decay_rate=.95,
            )
        total_iters = num_epochs * num_batches
        learning_rates = schedule(jnp.arange(total_iters))
        assert learning_rates[0] == 1.0, "Learning rate must start at 1."

        @jit
        def train_step(carry, input):
            self.unconstrained_params, rolling_stats = carry
            key, learning_rates = input

            def _minibatch_step(carry, args):
                self.unconstrained_params, rolling_stats = carry
                # emissions = next(sample_generator)
                lrate, emissions = args
                emissions = jnp.expand_dims(emissions, 0)

                # Compute the stats for this minibatch
                minibatch_stats = self.e_step(emissions)
                # Scale them up to whole dataset
                scaled_minibatch_stats = tree_map(lambda x: jnp.sum(x, axis=0) * scale, minibatch_stats)
                expected_lp = self.log_prior() + scaled_minibatch_stats.marginal_loglik
                # Take a rolling average
                rolling_stats = tree_map(lambda s0, s1: (1-lrate) * s0 + lrate * s1, rolling_stats, scaled_minibatch_stats)
                # Add a batch dimension and call M-step
                batched_rolling_stats = tree_map(lambda x: jnp.expand_dims(x, axis=0), rolling_stats)
                self.m_step(emissions, batched_rolling_stats)

                return (self.unconstrained_params, rolling_stats), expected_lp

            # ------------------------------------------------------------------

            (params, rolling_stats), minibatch_log_probs = lax.scan(
                _minibatch_step,
                (self.unconstrained_params, rolling_stats),
                (learning_rates, batch_emissions)
            )
            return (params, rolling_stats), minibatch_log_probs.mean()

        # Initialize rolling sufficient statistics with 0-arrays
        init_stats = self._zeros_like_suff_stats()

        # Run it!
        (params, _), log_probs = lax.scan(
            train_step,
            (self.unconstrained_params, init_stats),
            (jr.split(key, num_epochs), learning_rates.reshape(num_epochs, num_batches))
        )
        self.unconstrained_params = params
        return log_probs