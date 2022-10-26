from abc import abstractmethod
from copy import deepcopy
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, value_and_grad, vmap
from jax.tree_util import tree_map, tree_leaves
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.abstractions import SSM
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hmm.inference import compute_transition_probs
from dynamax.hmm.inference import hmm_filter
from dynamax.hmm.inference import hmm_posterior_mode
from dynamax.hmm.inference import hmm_smoother
from dynamax.hmm.inference import hmm_two_filter_smoother
from dynamax.optimize import run_sgd
from dynamax.parameters import ParameterProperties
from dynamax.utils import pytree_len, pytree_sum
from tqdm.auto import trange



class BaseHMM(SSM):

    def __init__(self, num_states):
        self.num_states = num_states

    # Three helper functions to compute the initial probabilities,
    # transition matrix (or matrices), and conditional log likelihoods.
    # These are the args to the HMM inference functions, and they can
    # be computed using the generic SSM initial_distribution(),
    # transition_distribution(), and emission_distribution() functions.
    def _compute_initial_probs(self, params, covariates=None):
        return self.initial_distribution(params, covariates).probs_parameter()

    def _compute_transition_matrices(self, params, covariates=None):
        if covariates is not None:
            f = lambda covariate: \
                vmap(lambda state: \
                    self.transition_distribution(params, state, covariate).probs_parameter())(
                        jnp.arange(self.num_states))
            next_covariates = tree_map(lambda x: x[1:], covariates)
            return vmap(f)(next_covariates)
        else:
            g = vmap(lambda state: self.transition_distribution(params, state).probs_parameter())
            return g(jnp.arange(self.num_states))

    def _compute_conditional_logliks(self, params, emissions, covariates=None):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, covariate: \
            vmap(lambda state: self.emission_distribution(params, state, covariate).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, covariates)

    # Basic inference code
    def marginal_log_prob(self, params, emissions, covariates=None):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self._compute_initial_probs(params, covariates),
                          self._compute_transition_matrices(params, covariates),
                          self._compute_conditional_logliks(params, emissions, covariates))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, params, emissions, covariates=None):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self._compute_initial_probs(params, covariates),
                                  self._compute_transition_matrices(params, covariates),
                                  self._compute_conditional_logliks(params, emissions, covariates))

    def filter(self, params, emissions, covariates=None):
        """Compute filtering distribution."""
        return hmm_filter(self._compute_initial_probs(params, covariates),
                          self._compute_transition_matrices(params, covariates),
                          self._compute_conditional_logliks(params, emissions, covariates))

    def smoother(self, params, emissions, covariates=None):
        """Compute smoothing distribution."""
        return hmm_smoother(self._compute_initial_probs(params, covariates),
                            self._compute_transition_matrices(params, covariates),
                            self._compute_conditional_logliks(params, emissions, covariates))

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, covariates=None):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        transition_matrices = self._compute_transition_matrices(params, covariates)
        posterior = hmm_two_filter_smoother(self._compute_initial_probs(params, covariates),
                                            transition_matrices,
                                            self._compute_conditional_logliks(params, emissions, covariates))

        # Compute expectations needed by the M-step
        initial_stats = posterior.initial_probs
        transition_stats = compute_transition_probs(transition_matrices, posterior, (transition_matrices.ndim == 2))
        emission_stats = posterior.smoothed_probs
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def m_step(self,
               curr_params,
               param_props,
               batch_emissions,
               batch_stats,
               batch_covariates=None,
               optimizer=optax.adam(1e-2),
               num_sgd_epochs_per_mstep=50):
        """_summary_
        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        curr_unc_params, fixed_params = to_unconstrained(curr_params, param_props)

        def neg_expected_log_joint(unc_params, minibatch):
            params = from_unconstrained(unc_params, fixed_params, param_props)

            minibatch_emissions, minibatch_stats, minibatch_covariates = minibatch
            scale = pytree_len(batch_emissions) / pytree_len(minibatch_emissions)

            def _single_expected_log_joint(emissions, stats, covariates):
                expected_initial_state, expected_transitions, expected_states = stats
                initial_probs = self._compute_initial_probs(params, covariates)
                trans_matrices = self._compute_transition_matrices(params, covariates)
                log_likelihoods = self._compute_conditional_logliks(params, emissions, covariates)

                expected_lp = jnp.sum(expected_initial_state * jnp.log(initial_probs))
                expected_lp += jnp.sum(expected_transitions * jnp.log(trans_matrices))
                expected_lp += jnp.sum(expected_states * log_likelihoods)
                return expected_lp

            log_prior = self.log_prior(params)
            minibatch_lps = vmap(_single_expected_log_joint)(
                minibatch_emissions, minibatch_stats, minibatch_covariates)
            expected_log_joint = log_prior + minibatch_lps.sum() * scale
            return -expected_log_joint / tree_leaves(batch_emissions)[0].size

        # Minimize the negative expected log joint with SGD
        unc_params, _ = run_sgd(neg_expected_log_joint,
                                curr_unc_params
                                (batch_emissions, batch_stats, batch_covariates),
                                optimizer=optimizer,
                                num_epochs=num_sgd_epochs_per_mstep)

        return from_unconstrained(unc_params, fixed_params, param_props)


class StandardHMM(BaseHMM):
    """The "Standard" HMM has an initial distribution $\pi_0$, a transition matrix $P$,
    and emission parameters $\{\theta_k\}_{k=1}^K$ that parameterize the emission
    distribution, $p(x \mid z_t=k, \theta_k)$

    Args:
        StandardHMM (_type_): _description_
    """

    def __init__(self,
                 num_states,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
        """
        Abstract base class for Hidden Markov Models.
        Child class specifies the emission distribution.
        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        super().__init__(num_states)

        # And the hyperparameters of the prior
        self.initial_probs_concentration = initial_probs_concentration * jnp.ones(num_states)
        self.transition_matrix_concentration = transition_matrix_concentration * jnp.ones(num_states)

    def initial_distribution(self, params, covariates=None):
        return tfd.Categorical(probs=params['initial']['probs'])

    def transition_distribution(self, params, state, covariates=None):
        return tfd.Categorical(probs=params['transitions']['transition_matrix'][state])

    def _compute_initial_probs(self, params, covariates=None):
        return params['initial']['probs']

    def _compute_transition_matrices(self, params, covariates=None):
        return params['transitions']['transition_matrix']

    # @abstractmethod
    # def _initialize_emissions(self, key, method="prior", **kwargs):
    #     """Initialize the emissions parameters

    #     Returns:
    #         params: nested dictionary of emission parameters
    #         props: matching nested dictionary emission parameter properties
    #     """

    @abstractmethod
    def initialize(self, key=None, method="prior", initial_probs=None, transition_matrix=None, **kwargs):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".
            initial_probs (_type_, optional): _description_. Defaults to None.
            transition_matrix (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Initialize the initial probabilities
        if initial_probs is None:
            this_key, key = jr.split(key)
            initial_probs = tfd.Dirichlet(self.initial_probs_concentration).sample(seed=this_key)
        else:
            assert initial_probs.shape == (self.num_states,)
            assert jnp.all(initial_probs >= 0)
            assert jnp.allclose(initial_probs.sum(), 1.0)

        # Initialize the transition matrix
        if transition_matrix is None:
            this_key, key = jr.split(key)
            transition_matrix = tfd.Dirichlet(self.transition_matrix_concentration)\
                .sample(seed=this_key, sample_shape=(self.num_states,))
        else:
            assert transition_matrix.shape == (self.num_states, self.num_states)
            assert jnp.all(transition_matrix >= 0)
            assert jnp.allclose(transition_matrix.sum(axis=1), 1.0)

        # Package the results into dictionaries
        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix))
        props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())))

        # Subclasses must overload this method and add 'emissions' parameters to these dicts
        return params, props

    @abstractmethod
    def emission_distribution(self, params, state, covariates=None):
        """Return a distribution over emissions given current state.
        Args:
            state (PyTree): current latent state.
        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        return lp

    def _m_step_initial_probs(self, params, param_props, initial_stats):
        if not param_props['initial']['probs'].trainable:
            return

        if self.num_states == 1:
            params['initial']['probs'] = jnp.array([1.0])
            return

        post = tfd.Dirichlet(self.initial_probs_concentration + initial_stats)
        params['initial']['probs'] = post.mode()
        return params

    def _m_step_transition_matrix(self, params, param_props, transition_stats):
        if not param_props['transitions']['transition_matrix'].trainable:
            return

        if self.num_states == 1:
            params['transitions']['transition_matrix'] = jnp.array([[1.0]])
            return

        post = tfd.Dirichlet(self.transition_matrix_concentration + transition_stats)
        params['transitions']['transition_matrix'] = post.mode()
        return params

    def _m_step_emissions(self,
                          curr_params,
                          param_props,
                          batch_emissions,
                          batch_emission_stats,
                          batch_covariates=None,
                          optimizer=optax.adam(1e-2),
                          num_mstep_iters=50):

        # freeze the initial and transition parameters
        temp_param_props = deepcopy(param_props)
        for props in temp_param_props['initial'].values():
            props.trainable = False
        for props in temp_param_props['transitions'].values():
            props.trainable = False

        # Extract the remaining unconstrained params, which should only be for the emissions.
        curr_unc_params, fixed_params = to_unconstrained(curr_params, temp_param_props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, temp_param_props)

            def _single_expected_log_like(emissions, expected_states, covariates):
                log_likelihoods = self._compute_conditional_logliks(params, emissions, covariates)
                lp = jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(
                batch_emissions, batch_emission_stats, batch_covariates)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / tree_leaves(batch_emissions)[0].size

        # Minimize the negative expected log joint with gradient descent
        loss_grad_fn = value_and_grad(neg_expected_log_joint)
        opt_state = optimizer.init(curr_unc_params)

        # One step of the algorithm
        def train_step(carry, args):
            params, opt_state = carry
            loss, grads = loss_grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        # Run the optimizer
        initial_carry =  (curr_unc_params, opt_state)
        (unc_params, _), losses = lax.scan(train_step, initial_carry, None, length=num_mstep_iters)

        # Return the updated parameters
        return from_unconstrained(unc_params, fixed_params, temp_param_props)

    def m_step(self,
               params,
               param_props,
               batch_emissions,
               batch_stats,
               batch_covariates=None,
               default_mstep_kwargs=dict(optimizer=optax.adam(1e-2), num_mstep_iters=50,)):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        params = self._m_step_initial_probs(params, param_props, batch_initial_stats.sum(0))
        params = self._m_step_transition_matrix(params, param_props, batch_transition_stats.sum(0))
        params = self._m_step_emissions(params, param_props, batch_emissions, batch_emission_stats,
                                        batch_covariates=batch_covariates, **default_mstep_kwargs)
        return params


class ExponentialFamilyHMM(StandardHMM):
    """
    These models belong the exponential family of distributions and return a
    set of expected sufficient statistics instead of an HMMPosterior object.
    """
    @abstractmethod
    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        raise NotImplementedError

    @abstractmethod
    def _zeros_like_suff_stats(self):
        raise NotImplementedError

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, covariates=None):
        """For exponential family emissions, the E step returns the sum of expected
        sufficient statistics rather than the expected states for each time step.
        """
        (initial_stats, transition_stats, expected_states), ll = \
            super().e_step(params, emissions, covariates)
        emission_stats = self._compute_expected_suff_stats(params, emissions, expected_states, covariates)
        return (initial_stats, transition_stats, emission_stats), ll

    @abstractmethod
    def _m_step_emissions(self, params, param_props, emission_stats):
        raise NotImplementedError

    def m_step(self,
               params,
               param_props,
               batch_emissions,
               batch_stats,
               batch_covariates=None):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        params = self._m_step_initial_probs(params, param_props, batch_initial_stats.sum(0))
        params = self._m_step_transition_matrix(params, param_props, batch_transition_stats.sum(0))
        params = self._m_step_emissions(params, param_props, pytree_sum(batch_emission_stats, axis=0))
        return params

    def fit_stochastic_em(self, initial_params, param_props, emissions_generator, schedule=None, num_epochs=50):
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
            emissions_generator: Iterable over the emissions dataset;
                auto-shuffles batches after each epoch.
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
            # TODO: Handle minibatch covariates
            minibatch_stats, lls = vmap(partial(self.e_step, params))(minibatch_emissions)
            # minibatch_stats, ll = self.e_step(params, minibatch_emissions)

            # Scale the stats as if they came from the whole dataset
            scale = num_batches
            scaled_minibatch_stats = tree_map(lambda x: jnp.sum(x, axis=0) * scale, minibatch_stats)
            expected_lp = self.log_prior(params) + lls.sum() * scale

            # Incorporate these these stats into the rolling averaged stats
            rolling_stats = tree_map(lambda s0, s1: (1 - learn_rate) * s0 + learn_rate * s1, rolling_stats,
                                     scaled_minibatch_stats)

            # Add a batch dimension and call M-step
            batched_rolling_stats = tree_map(lambda x: jnp.expand_dims(x, axis=0), rolling_stats)
            params = self.m_step(params, param_props, minibatch_emissions, batched_rolling_stats)

            return (params, rolling_stats), expected_lp

        # Initialize and train
        params = initial_params
        expected_log_probs = []
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
        return params, jnp.array(expected_log_probs)
