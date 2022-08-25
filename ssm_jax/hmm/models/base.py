from abc import abstractmethod
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import jit
from jax import lax
from jax import tree_leaves
from jax import tree_map
from jax import vmap

from jax import vmap
from jax.tree_util import tree_map
from ssm_jax.abstractions import SSM
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.optimize import run_sgd
from tqdm.auto import trange


class BaseHMM(SSM):

    # Properties to get various attributes of the model.
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
    def _compute_initial_probs(self, **covariates):
        return self.initial_distribution(**covariates).probs_parameter()

    def _compute_transition_matrices(self, **covariates):
        if len(covariates) > 0:
            f = lambda **covariate: \
                vmap(lambda state: \
                    self.transition_distribution(state, **covariate).probs_parameter())(
                        jnp.arange(self.num_states))
            next_covariates = tree_map(lambda x: x[1:], covariates)
            return vmap(f)(**next_covariates)
        else:
            g = vmap(lambda state: self.transition_distribution(state).probs_parameter())
            return g(jnp.arange(self.num_states))

    def _compute_conditional_logliks(self, emissions, **covariates):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, **covariate: \
            vmap(lambda state: self.emission_distribution(state, **covariate).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, **covariates)

    # Basic inference code
    def marginal_log_prob(self, emissions, **covariates):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self._compute_initial_probs(**covariates),
                          self._compute_transition_matrices(**covariates),
                          self._compute_conditional_logliks(emissions, **covariates))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions, **covariates):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self._compute_initial_probs(**covariates),
                                  self._compute_transition_matrices(**covariates),
                                  self._compute_conditional_logliks(emissions, **covariates))

    def filter(self, emissions, **covariates):
        """Compute filtering distribution."""
        return hmm_filter(self._compute_initial_probs(**covariates),
                          self._compute_transition_matrices(**covariates),
                          self._compute_conditional_logliks(emissions, **covariates))

    def smoother(self, emissions, **covariates):
        """Compute smoothing distribution."""
        return hmm_smoother(self._compute_initial_probs(**covariates),
                            self._compute_transition_matrices(**covariates),
                            self._compute_conditional_logliks(emissions, **covariates))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions, **batch_covariates):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        def _single_e_step(emissions, **covariates):
            transition_matrices = self._compute_transition_matrices(**covariates)
            posterior = hmm_two_filter_smoother(self._compute_initial_probs(**covariates),
                                                transition_matrices,
                                                self._compute_conditional_logliks(emissions, **covariates))

            # Compute the transition probabilities
            posterior.trans_probs = compute_transition_probs(transition_matrices,
                                                             posterior,
                                                             reduce_sum=(transition_matrices.ndim == 2))

            return posterior

        return vmap(_single_e_step)(batch_emissions, **batch_covariates)

    def m_step(self,
               batch_emissions,
               batch_posteriors,
               optimizer=optax.adam(1e-2),
               num_sgd_epochs_per_mstep=50,
               **batch_covariates):
        """_summary_
        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """

        def neg_expected_log_joint(params, minibatch):
            minibatch_emissions, minibatch_posteriors, minibatch_covariates = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            self.unconstrained_params = params

            def _single_expected_log_joint(emissions, posterior, **covariates):
                initial_probs = self._compute_initial_probs(**covariates)
                trans_matrices = self._compute_transition_matrices(**covariates)
                log_likelihoods = self._compute_conditional_logliks(emissions, **covariates)
                expected_states = posterior.smoothed_probs
                trans_probs = posterior.trans_probs

                lp = jnp.sum(expected_states[0] * jnp.log(initial_probs))
                lp += jnp.sum(trans_probs * jnp.log(trans_matrices))
                lp += jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior()
            minibatch_lps = vmap(_single_expected_log_joint)(
                minibatch_emissions, minibatch_posteriors, **minibatch_covariates)
            expected_log_joint = log_prior + minibatch_lps.sum() * scale
            return -expected_log_joint / batch_emissions.size

        # Minimize the negative expected log joint with SGD
        params, losses = run_sgd(neg_expected_log_joint,
                                 self.unconstrained_params,
                                 (batch_emissions, batch_posteriors, batch_covariates),
                                 optimizer=optimizer,
                                 num_epochs=num_sgd_epochs_per_mstep)
        self.unconstrained_params = params

    def fit_em(self, batch_emissions, num_iters=50, mstep_kwargs=dict(), **batch_covariates):
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
            batch_posteriors = self.e_step(batch_emissions, **batch_covariates)
            lp = self.log_prior() + batch_posteriors.marginal_loglik.sum()
            self.m_step(batch_emissions, batch_posteriors, **mstep_kwargs, **batch_covariates)
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
                **batch_covariates):
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
        def _loss_fn(params, minibatch_emissions, **minibatch_covariates):
            """Default objective function."""
            self.unconstrained_params = params
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(self.marginal_log_prob)(minibatch_emissions, **minibatch_covariates)
            lp = self.log_prior() + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        params, losses = run_sgd(_loss_fn,
                                 self.unconstrained_params,
                                 batch_emissions,
                                 optimizer=optimizer,
                                 batch_size=batch_size,
                                 num_epochs=num_epochs,
                                 shuffle=shuffle,
                                 key=key,
                                 **batch_covariates)
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

    def initial_distribution(self, **covariates):
        return tfd.Categorical(probs=self._initial_probs.value)

    def transition_distribution(self, state, **covariates):
        return tfd.Categorical(probs=self._transition_matrix.value[state])

    def _compute_initial_probs(self, **covariates):
        return self.initial_probs.value

    def _compute_transition_matrices(self, **covariates):
        return self.transition_matrix.value

    @abstractmethod
    def emission_distribution(self, state, **covariates):
        """Return a distribution over emissions given current state.
        Args:
            state (PyTree): current latent state.
        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    def _m_step_initial_probs(self, batch_emissions, batch_posteriors, **batch_covariates):
        post = tfd.Dirichlet(self._initial_probs_concentration.value + batch_posteriors.initial_probs.sum(axis=0))
        self._initial_probs.value = post.mode()

    def _m_step_transition_matrix(self, batch_emissions, batch_posteriors, **batch_covariates):
        post = tfd.Dirichlet(self._transition_matrix_concentration.value + batch_posteriors.trans_probs.sum(axis=0))
        self._transition_matrix.value = post.mode()

    def _m_step_emissions(self,
                          batch_emissions,
                          batch_posteriors,
                          optimizer=optax.adam(1e-2),
                          num_mstep_iters=50,
                          **batch_covariates):

        def neg_expected_log_joint(params, minibatch):
            minibatch_emissions, minibatch_posteriors, minibatch_covariates = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            self.unconstrained_params = params

            def _single_expected_log_like(emissions, posterior, **covariates):
                log_likelihoods = self._compute_conditional_logliks(emissions, **covariates)
                expected_states = posterior.smoothed_probs
                lp = 0.0
                lp += jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior()
            minibatch_ells = vmap(_single_expected_log_like)(
                minibatch_emissions, minibatch_posteriors, **minibatch_covariates)
            expected_log_joint = log_prior + minibatch_ells.sum() * scale
            return -expected_log_joint / batch_emissions.size

        # Minimize the negative expected log joint with SGD
        params, losses = run_sgd(neg_expected_log_joint,
                                 self.unconstrained_params,
                                 (batch_emissions, batch_posteriors, batch_covariates),
                                 optimizer=optimizer,
                                 num_epochs=num_mstep_iters)
        self.unconstrained_params = params

    def m_step(self,
               batch_emissions,
               batch_posteriors,
               generic_mstep_kwargs=dict(
                   optimizer=optax.adam(1e-2),
                   num_mstep_iters=50,
               ),
               **batch_covariates):

        is_initial_probs_frozen = self.initial_probs.is_frozen
        is_transition_matrix_frozen = self.transition_matrix.is_frozen

        self.initial_probs.freeze()
        self.transition_matrix.freeze()

        self._m_step_emissions(batch_emissions, batch_posteriors, **generic_mstep_kwargs, **batch_covariates)

        if not is_initial_probs_frozen:
            self.initial_probs.unfreeze()
            self._m_step_initial_probs(batch_emissions, batch_posteriors, **batch_covariates)

        if not is_transition_matrix_frozen:
            self.transition_matrix.unfreeze()
            self._m_step_transition_matrix(batch_emissions, batch_posteriors, **batch_covariates)


class ExponentialFamilyHMM(StandardHMM):
    """
    These models belong the exponential family of distributions and return a
    set of expected sufficient statistics instead of an HMMPosterior object.
    """

    @abstractmethod
    def _zeros_like_suff_stats(self):
        raise NotImplementedError

    def fit_stochastic_em(self, emissions_generator, total_emissions, schedule=None, num_epochs=50):
        """
        Fit this HMM by running Stochastic Expectation-Maximization.
        Assuming the original dataset consists of N independent sequences of
        length T, this algorithm performs EM on a random subset of B sequences
        (not timesteps) at each step. Importantly, the subsets of B sequences
        are shuffled at each epoch. It is up to the user to correctly
        instantiate the Dataloader generator object to exhibit this property.
        The algorithm uses a learning rate schedule to anneal the minibatch
        sufficient statistics at each stage of training. If a schedule is not
        specified, an exponentially decaying model is used such that the
        learning rate which decreases by 5% at each epoch.

        Args:
            emissions_generator (torch.data.utils.Dataloader): Iterable over the
                emissions dataset; auto-shuffles batches after each epoch.
            total_emissions (int): Total number of emissions that the generator
                will load. Used to scale the minibatch statistics.
            schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
                schedule; defaults to exponential schedule.
            num_epochs (int): Num of iterations made through the entire dataset.
        Returns:
            expected_log_prob (chex.Array): Mean expected log prob of each epoch.

        TODO Any way to take a weighted average of rolling stats (in addition
             to the convex combination) given the number of emissions we see
             with each new minibatch? This would allow us to remove the
             `total_emissions` variable, and avoid errors in math in calculating
             total number of emissions (which could get tricky esp. with
             variable batch sizes.)
        """

        num_batches = len(emissions_generator)

        # Set global training learning rates: shape (num_epochs, num_batches)
        if schedule is None:
            schedule = optax.exponential_decay(
                init_value=1.,
                end_value=0.,
                transition_steps=num_batches,
                decay_rate=.95,
            )

        learning_rates = schedule(jnp.arange(num_epochs * num_batches))
        assert learning_rates[0] == 1.0, "Learning rate must start at 1."
        learning_rates = learning_rates.reshape(num_epochs, num_batches)

        @jit
        def minibatch_em_step(carry, inputs):
            params, rolling_stats = carry
            minibatch_emissions, learn_rate = inputs

            # Compute the sufficient stats given a minibatch of emissions
            self.unconstrained_params = params
            minibatch_stats = self.e_step(minibatch_emissions)

            # Scale the stats as if they came from the whole dataset
            scale = total_emissions / len(minibatch_emissions.reshape(-1, self.num_obs))
            scaled_minibatch_stats = tree_map(lambda x: jnp.sum(x, axis=0) * scale, minibatch_stats)
            expected_lp = self.log_prior() + scaled_minibatch_stats.marginal_loglik

            # Incorporate these these stats into the rolling averaged stats
            rolling_stats = tree_map(lambda s0, s1: (1 - learn_rate) * s0 + learn_rate * s1, rolling_stats,
                                     scaled_minibatch_stats)

            # Add a batch dimension and call M-step
            batched_rolling_stats = tree_map(lambda x: jnp.expand_dims(x, axis=0), rolling_stats)
            self.m_step(minibatch_emissions, batched_rolling_stats)

            return (self.unconstrained_params, rolling_stats), expected_lp

        # Initialize and train
        expected_log_probs = []
        params = self.unconstrained_params
        rolling_stats = self._zeros_like_suff_stats()
        for epoch in trange(num_epochs):

            _expected_lps = 0.
            for minibatch, minibatch_emissions in enumerate(emissions_generator):
                (params, rolling_stats), expected_lp = minibatch_em_step(
                    (params, rolling_stats),
                    (minibatch_emissions, learning_rates[epoch][minibatch]),
                )
                _expected_lps += expected_lp

            # Save epoch mean of expected log probs
            expected_log_probs.append(_expected_lps / num_batches)

        # Update self with fitted params
        self.unconstrained_params = params
        return jnp.array(expected_log_probs)
