from abc import abstractmethod, ABC
from dynamax.abstractions import SSM
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hmm.inference import compute_transition_probs
from dynamax.hmm.inference import hmm_filter
from dynamax.hmm.inference import hmm_posterior_mode
from dynamax.hmm.inference import hmm_smoother
from dynamax.hmm.inference import hmm_two_filter_smoother
from dynamax.utils import pytree_len, pytree_slice
import jax.numpy as jnp
from jax import lax, value_and_grad, vmap
from jax.tree_util import tree_map
import optax


class HMMInitialState(ABC):
    """Abstract class for HMM initial distributions.
    """
    @abstractmethod
    def distribution(self, params, covariates=None):
        """Return a distribution over the initial latent state

        Returns:
            dist (tfd.Distribution): conditional distribution of initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, key=None, method="prior", **kwargs):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".

        Returns:
            params
            props
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params):
        raise NotImplementedError

    def compute_initial_probs(self, params, covariates=None):
        return self.initial_distribution(params, covariates).probs_parameter()

    def collect_suff_stats(self, posterior, covariates=None):
        return posterior.smoothed_probs[0], pytree_slice(covariates, 0)

    def m_step(self,
               curr_params,
               param_props,
               batch_stats,
               optimizer=optax.adam(1e-2),
               num_mstep_iters=50):

        # Extract the remaining unconstrained params, which should only be for the emissions.
        curr_unc_params, fixed_params = to_unconstrained(curr_params, param_props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, param_props)

            def _single_expected_log_like(stats):
                smoothed_initial_prob, covariate = stats
                log_initial_prob = jnp.log(self.compute_initial_probs(params, covariate))
                lp = jnp.sum(smoothed_initial_prob * log_initial_prob)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / pytree_len(batch_stats)

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
        return from_unconstrained(unc_params, fixed_params, param_props)


class HMMTransitions(ABC):
    """Abstract class for HMM transitions.
    """
    @abstractmethod
    def distribution(self, params, state, covariates=None):
        """Return a distribution over the next state given the current state.

        Args:
            state (PyTree): current latent state.
        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, key=None, method="prior", **kwargs):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".

        Returns:
            params
            props
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params):
        raise NotImplementedError

    def compute_transition_matrices(self, params, covariates=None):
        if covariates is not None:
            f = lambda covariate: \
                vmap(lambda state: \
                    self.distribution(params, state, covariate).probs_parameter())(
                        jnp.arange(self.num_states))
            next_covariates = tree_map(lambda x: x[1:], covariates)
            return vmap(f)(next_covariates)
        else:
            g = vmap(lambda state: self.distribution(params, state).probs_parameter())
            return g(jnp.arange(self.num_states))

    def collect_suff_stats(self, posterior, covariates=None):
        return posterior.trans_probs, pytree_slice(covariates, slice(1, None))

    def m_step(self,
               curr_params,
               param_props,
               batch_transition_stats,
               batch_covariates=None,
               optimizer=optax.adam(1e-2),
               num_mstep_iters=50,
               scale=1.0):

        curr_unc_params, fixed_params = to_unconstrained(curr_params, param_props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, param_props)

            def _single_expected_log_like(expected_transitions, covariates):
                log_trans_matrix = jnp.log(self.compute_transition_matrices(params, covariates))
                lp = jnp.sum(expected_transitions * log_trans_matrix)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_transition_stats, batch_covariates)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

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
        return from_unconstrained(unc_params, fixed_params, param_props)


class HMMEmissions(ABC):
    """Abstract class for HMM emissions.
    """
    @property
    @abstractmethod
    def emission_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's emissions.
        For example, a Gaussian HMM with D dimensional emissions would return (D,).
        """
        raise NotImplementedError

    @abstractmethod
    def distribution(self, params, state, covariates=None):
        """Return a distribution over emissions given current state.

        Args:
            state (PyTree): current latent state.
        Returns:
            dist (tfd.Distribution): conditional distribution of current emission.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, key=None, method="prior", **kwargs):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to "prior".

        Returns:
            params
            props
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params):
        raise NotImplementedError

    def compute_conditional_logliks(self, params, emissions, covariates=None):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, covariate: \
            vmap(lambda state: self.distribution(params, state, covariate).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, covariates)

    def collect_suff_stats(self, posterior, emissions, covariates=None):
        return posterior.smoothed_probs, emissions, covariates

    def m_step(self, params, props, batch_stats,
               optimizer=optax.adam(1e-2), num_mstep_iters=50, scale=1.0):

        # Extract the remaining unconstrained params, which should only be for the emissions.
        curr_unc_params, fixed_params = to_unconstrained(params, props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, props)

            def _single_expected_log_like(emissions, stats):
                expected_states, emissions, covariates = stats
                log_likelihoods = self.compute_conditional_logliks(params, emissions, covariates)
                lp = jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

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
        return from_unconstrained(unc_params, fixed_params, props)


class HMM(SSM):

    def __init__(self,
                 num_states : int,
                 initial_component : HMMInitialState,
                 transition_component : HMMTransitions,
                 emission_component : HMMEmissions):
        """Abstract base class for HMMs.

        Args:
            num_states (_type_): _description_
            initial_component (_type_): _description_
            transition_component (_type_): _description_
            emission_component (_type_): _description_
        """

        self.num_states = num_states
        self.initial_component = initial_component
        self.transition_component = transition_component
        self.emission_component = emission_component

    # Implement the SSM abstract methods by passing on to the components
    @property
    def emission_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's emissions.
        For example, a Gaussian HMM with D dimensional emissions would return (D,).
        """
        return self.emission_component.emission_shape

    def initial_distribution(self, params, covariates=None):
        return self.initial_component.distribution(params["initial"], covariates=covariates)

    def transition_distribution(self, params, state, covariates=None):
        return self.transition_component.distribution(params["transitions"], state, covariates=covariates)

    def emission_distribution(self, params, state, covariates=None):
        return self.emission_component.distribution(params["emissions"], state, covariates=covariates)

    def log_prior(self, params):
        lp = self.initial_component.log_prior(params["initial"])
        lp += self.transition_component.log_prior(params["transitions"])
        lp += self.emission_component.log_prior(params["emissions"])
        return lp

    # The inference functions all need the same arguments
    def _inference_args(self, params, emissions, covariates):
        return (self.initial_component.compute_initial_probs(params["initial"], covariates),
                self.transition_component.compute_transition_matrices(params["transitions"], covariates),
                self.emission_component.compute_conditional_logliks(params["emissions"], emissions, covariates))

    # Convenience wrappers for the inference code
    def marginal_log_prob(self, params, emissions, covariates=None):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(*self._inference_args(params, emissions, covariates))
        return post.marginal_loglik

    def most_likely_states(self, params, emissions, covariates=None):
        """Compute most likely state path with the Viterbi algorithm."""
        return hmm_posterior_mode(*self._inference_args(params, emissions, covariates))

    def filter(self, params, emissions, covariates=None):
        """Compute filtering distribution."""
        return hmm_filter(*self._inference_args(params, emissions, covariates))

    def smoother(self, params, emissions, covariates=None):
        """Compute smoothing distribution."""
        return hmm_smoother(*self._inference_args(params, emissions, covariates))

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, covariates=None):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        args = self._inference_args(params, emissions, covariates)
        posterior = hmm_two_filter_smoother(*args)
        posterior.trans_probs = compute_transition_probs(args[1], posterior, (args[1].ndim == 2))

        initial_stats = self.initial_component.collect_suff_stats(posterior, covariates)
        transition_stats = self.transition_component.collect_suff_stats(posterior, covariates)
        emission_stats = self.emission_component.collect_suff_stats(posterior, emissions, covariates)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def m_step(self, params, props, batch_stats):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        params["initial"] = self.initial_component.m_step(params["initial"], props["initial"], batch_initial_stats)
        params["transitions"] = self.transition_component.m_step(params["transitions"], props["transitions"], batch_transition_stats)
        params["emissions"] = self.emission_component.m_step(params["emissions"], props["emissions"], batch_emission_stats)
        return params


# class ExponentialFamilyHMM(StandardHMM):
#     """
#     These models belong the exponential family of distributions and return a
#     set of expected sufficient statistics instead of an HMMPosterior object.
#     """
#     @abstractmethod
#     def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
#         raise NotImplementedError

#     @abstractmethod
#     def _zeros_like_suff_stats(self):
#         raise NotImplementedError

#     # Expectation-maximization (EM) code
#     def e_step(self, params, emissions, covariates=None):
#         """For exponential family emissions, the E step returns the sum of expected
#         sufficient statistics rather than the expected states for each time step.
#         """
#         (initial_stats, transition_stats, expected_states), ll = \
#             super().e_step(params, emissions, covariates)
#         emission_stats = self._compute_expected_suff_stats(params, emissions, expected_states, covariates)
#         return (initial_stats, transition_stats, emission_stats), ll

#     @abstractmethod
#     def _m_step_emissions(self, params, param_props, emission_stats):
#         raise NotImplementedError

#     def fit_stochastic_em(self, initial_params, param_props, emissions_generator, schedule=None, num_epochs=50):
#         """
#         Fit this HMM by running Stochastic Expectation-Maximization.
#         Assuming the original dataset consists of N independent sequences of
#         length T, this algorithm performs EM on a random subset of B sequences
#         (not timesteps) at each step. Importantly, the subsets of B sequences
#         are shuffled at each epoch. It is up to the user to correctly
#         instantiate the Dataloader generator object to exhibit this property.
#         The algorithm uses a learning rate schedule to anneal the minibatch
#         sufficient statistics at each stage of training. If a schedule is not
#         specified, an exponentially decaying model is used such that the
#         learning rate which decreases by 5% at each epoch.

#         Args:
#             emissions_generator: Iterable over the emissions dataset;
#                 auto-shuffles batches after each epoch.
#             total_emissions (int): Total number of emissions that the generator
#                 will load. Used to scale the minibatch statistics.
#             schedule (optax schedule, Callable: int -> [0, 1]): Learning rate
#                 schedule; defaults to exponential schedule.
#             num_epochs (int): Num of iterations made through the entire dataset.
#         Returns:
#             expected_log_prob (chex.Array): Mean expected log prob of each epoch.

#         TODO Any way to take a weighted average of rolling stats (in addition
#              to the convex combination) given the number of emissions we see
#              with each new minibatch? This would allow us to remove the
#              `total_emissions` variable, and avoid errors in math in calculating
#              total number of emissions (which could get tricky esp. with
#              variable batch sizes.)
#         """
#         num_batches = len(emissions_generator)

#         # Set global training learning rates: shape (num_epochs, num_batches)
#         if schedule is None:
#             schedule = optax.exponential_decay(
#                 init_value=1.,
#                 end_value=0.,
#                 transition_steps=num_batches,
#                 decay_rate=.95,
#             )

#         learning_rates = schedule(jnp.arange(num_epochs * num_batches))
#         assert learning_rates[0] == 1.0, "Learning rate must start at 1."
#         learning_rates = learning_rates.reshape(num_epochs, num_batches)

#         @jit
#         def minibatch_em_step(carry, inputs):
#             params, rolling_stats = carry
#             minibatch_emissions, learn_rate = inputs

#             # Compute the sufficient stats given a minibatch of emissions
#             # TODO: Handle minibatch covariates
#             minibatch_stats, lls = vmap(partial(self.e_step, params))(minibatch_emissions)
#             # minibatch_stats, ll = self.e_step(params, minibatch_emissions)

#             # Scale the stats as if they came from the whole dataset
#             scale = num_batches
#             scaled_minibatch_stats = tree_map(lambda x: jnp.sum(x, axis=0) * scale, minibatch_stats)
#             expected_lp = self.log_prior(params) + lls.sum() * scale

#             # Incorporate these these stats into the rolling averaged stats
#             rolling_stats = tree_map(lambda s0, s1: (1 - learn_rate) * s0 + learn_rate * s1, rolling_stats,
#                                      scaled_minibatch_stats)

#             # Add a batch dimension and call M-step
#             batched_rolling_stats = tree_map(lambda x: jnp.expand_dims(x, axis=0), rolling_stats)
#             params = self.m_step(params, param_props, minibatch_emissions, batched_rolling_stats)

#             return (params, rolling_stats), expected_lp

#         # Initialize and train
#         params = initial_params
#         expected_log_probs = []
#         rolling_stats = self._zeros_like_suff_stats()
#         for epoch in trange(num_epochs):

#             _expected_lps = 0.
#             for minibatch, minibatch_emissions in enumerate(emissions_generator):
#                 (params, rolling_stats), expected_lp = minibatch_em_step(
#                     (params, rolling_stats),
#                     (minibatch_emissions, learning_rates[epoch][minibatch]),
#                 )
#                 _expected_lps += expected_lp

#             # Save epoch mean of expected log probs
#             expected_log_probs.append(_expected_lps / num_batches)

#         # Update self with fitted params
#         return params, jnp.array(expected_log_probs)
