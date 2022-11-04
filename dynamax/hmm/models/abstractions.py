from abc import abstractmethod, ABC
from dynamax.abstractions import SSM
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.hmm.inference import compute_transition_probs
from dynamax.hmm.inference import hmm_filter
from dynamax.hmm.inference import hmm_posterior_mode
from dynamax.hmm.inference import hmm_smoother
from dynamax.hmm.inference import hmm_two_filter_smoother
from dynamax.optimize import run_gradient_descent
from dynamax.utils import pytree_slice
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
import optax


class HMMInitialState(ABC):
    """Abstract class for HMM initial distributions.
    """
    def __init__(self,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50) -> None:
        self.m_step_optimizer = m_step_optimizer
        self.m_step_num_iters = m_step_num_iters

    @abstractmethod
    def distribution(self, params, inputs=None):
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

    def compute_initial_probs(self, params, inputs=None):
        return self.initial_distribution(params, inputs).probs_parameter()

    def collect_suff_stats(self, params, posterior, inputs=None):
        return posterior.smoothed_probs[0], pytree_slice(inputs, 0)

    def initialize_m_step_state(self, params, props):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params, _ = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self, params, props, batch_stats, m_step_state, scale=1.0):

        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params, fixed_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, props)

            def _single_expected_log_like(stats):
                expected_initial_state, inpt = stats
                log_initial_prob = jnp.log(self.compute_initial_probs(params, inpt))
                lp = jnp.sum(expected_initial_state * log_initial_prob)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

        # Run gradient descent
        unc_params, m_step_state, losses = \
            run_gradient_descent(neg_expected_log_joint,
                                 unc_params,
                                 self.m_step_optimizer,
                                 optimizer_state=m_step_state,
                                 num_mstep_iters=self.m_step_num_iters)

        # Return the updated parameters and optimizer state
        params = from_unconstrained(unc_params, fixed_params, props)
        return params, m_step_state


class HMMTransitions(ABC):
    """Abstract class for HMM transitions.
    """
    def __init__(self,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50) -> None:
        self.m_step_optimizer = m_step_optimizer
        self.m_step_num_iters = m_step_num_iters

    @abstractmethod
    def distribution(self, params, state, inputs=None):
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

    def compute_transition_matrices(self, params, inputs=None):
        if inputs is not None:
            f = lambda inpt: \
                vmap(lambda state: \
                    self.distribution(params, state, inpt).probs_parameter())(
                        jnp.arange(self.num_states))
            next_inputs = tree_map(lambda x: x[1:], inputs)
            return vmap(f)(next_inputs)
        else:
            g = vmap(lambda state: self.distribution(params, state).probs_parameter())
            return g(jnp.arange(self.num_states))

    def collect_suff_stats(self, params, posterior, inputs=None):
        return posterior.trans_probs, pytree_slice(inputs, slice(1, None))

    def initialize_m_step_state(self, params, props):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params, _ = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self, params, props, batch_stats, m_step_state, scale=1.0):
        unc_params, fixed_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, props)

            def _single_expected_log_like(stats):
                expected_transitions, inputs = stats
                log_trans_matrix = jnp.log(self.compute_transition_matrices(params, inputs))
                lp = jnp.sum(expected_transitions * log_trans_matrix)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

        # Run gradient descent
        unc_params, m_step_state, losses = \
            run_gradient_descent(neg_expected_log_joint,
                                 unc_params,
                                 self.m_step_optimizer,
                                 optimizer_state=m_step_state,
                                 num_mstep_iters=self.m_step_num_iters)

        # Return the updated parameters and optimizer state
        params = from_unconstrained(unc_params, fixed_params, props)
        return params, m_step_state


class HMMEmissions(ABC):
    """Abstract class for HMM emissions.
    """
    def __init__(self,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50) -> None:
        self.m_step_optimizer = m_step_optimizer
        self.m_step_num_iters = m_step_num_iters

    @property
    @abstractmethod
    def emission_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's emissions.
        For example, a Gaussian HMM with D dimensional emissions would return (D,).
        """
        raise NotImplementedError

    @abstractmethod
    def distribution(self, params, state, inputs=None):
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

    def compute_conditional_logliks(self, params, emissions, inputs=None):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, inpt: \
            vmap(lambda state: self.distribution(params, state, inpt).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, inputs)

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        return posterior.smoothed_probs, emissions, inputs

    def initialize_m_step_state(self, params, props):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params, _ = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self, params, props, batch_stats, m_step_state, scale=1.0):

        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params, fixed_params = to_unconstrained(params, props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, fixed_params, props)

            def _single_expected_log_like(stats):
                expected_states, emissions, inputs = stats
                log_likelihoods = self.compute_conditional_logliks(params, emissions, inputs)
                lp = jnp.sum(expected_states * log_likelihoods)
                return lp

            log_prior = self.log_prior(params)
            batch_ells = vmap(_single_expected_log_like)(batch_stats)
            expected_log_joint = log_prior + batch_ells.sum()
            return -expected_log_joint / scale

        # Run gradient descent
        unc_params, m_step_state, losses = \
            run_gradient_descent(neg_expected_log_joint,
                                 unc_params,
                                 self.m_step_optimizer,
                                 optimizer_state=m_step_state,
                                 num_mstep_iters=self.m_step_num_iters)

        # Return the updated parameters and optimizer state
        params = from_unconstrained(unc_params, fixed_params, props)
        return params, m_step_state


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

    def initial_distribution(self, params, inputs=None):
        return self.initial_component.distribution(params["initial"], inputs=inputs)

    def transition_distribution(self, params, state, inputs=None):
        return self.transition_component.distribution(params["transitions"], state, inputs=inputs)

    def emission_distribution(self, params, state, inputs=None):
        return self.emission_component.distribution(params["emissions"], state, inputs=inputs)

    def log_prior(self, params):
        lp = self.initial_component.log_prior(params["initial"])
        lp += self.transition_component.log_prior(params["transitions"])
        lp += self.emission_component.log_prior(params["emissions"])
        return lp

    # The inference functions all need the same arguments
    def _inference_args(self, params, emissions, inputs):
        return (self.initial_component.compute_initial_probs(params["initial"], inputs),
                self.transition_component.compute_transition_matrices(params["transitions"], inputs),
                self.emission_component.compute_conditional_logliks(params["emissions"], emissions, inputs))

    # Convenience wrappers for the inference code
    def marginal_log_prob(self, params, emissions, inputs=None):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(*self._inference_args(params, emissions, inputs))
        return post.marginal_loglik

    def most_likely_states(self, params, emissions, inputs=None):
        """Compute most likely state path with the Viterbi algorithm."""
        return hmm_posterior_mode(*self._inference_args(params, emissions, inputs))

    def filter(self, params, emissions, inputs=None):
        """Compute filtering distribution."""
        return hmm_filter(*self._inference_args(params, emissions, inputs))

    def smoother(self, params, emissions, inputs=None):
        """Compute smoothing distribution."""
        return hmm_smoother(*self._inference_args(params, emissions, inputs))

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, inputs=None):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args)
        posterior.trans_probs = compute_transition_probs(args[1], posterior, (args[1].ndim == 2))

        initial_stats = self.initial_component.collect_suff_stats(params["initial"], posterior, inputs)
        transition_stats = self.transition_component.collect_suff_stats(params["transitions"], posterior, inputs)
        emission_stats = self.emission_component.collect_suff_stats(params["emissions"], posterior, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def initialize_m_step_state(self, params, props):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        initial_m_step_state = self.initial_component.initialize_m_step_state(params["initial"], props["initial"])
        transitions_m_step_state = self.transition_component.initialize_m_step_state(params["transitions"], props["transitions"])
        emissions_m_step_state = self.emission_component.initialize_m_step_state(params["emissions"], props["emissions"])
        return initial_m_step_state, transitions_m_step_state, emissions_m_step_state

    def m_step(self, params, props, batch_stats, m_step_state):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state
        params.initial, initial_m_step_state = self.initial_component.m_step(params.initial, props["initial"], batch_initial_stats, initial_m_step_state)
        params.transitions, transitions_m_step_state = self.transition_component.m_step(params.transitions, props["transitions"], batch_transition_stats, transitions_m_step_state)
        params.emissions, emissions_m_step_state = self.emission_component.m_step(params.emissions, props["emissions"], batch_emission_stats, emissions_m_step_state)
        m_step_state = initial_m_step_state, transitions_m_step_state, emissions_m_step_state
        return params, m_step_state


# class ExponentialFamilyHMM(HMM):
#     """
#     An HMM whose initial distribution, transition distribution, and emission
#     distribution all belong to the exponential family. Such models admit a
#     simple stochastic expectation-maximization algorithm.
#     """
#     def fit_stochastic_em(self,
#                           initial_params,
#                           param_props,
#                           emissions_generator,
#                           schedule=None,
#                           num_epochs=50):
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
#             # TODO: Handle minibatch inputs
#             minibatch_stats, lls = vmap(partial(self.e_step, params))(minibatch_emissions)
#             # minibatch_stats, ll = self.e_step(params, minibatch_emissions)

#             # Scale the stats as if they came from the whole dataset
#             scale = num_batches
#             scaled_minibatch_stats = tree_map(lambda x: jnp.sum(x, axis=0) * scale, minibatch_stats)
#             expected_lp = self.log_prior(params) + lls.sum() * scale

#             # Incorporate these these stats into the rolling averaged stats
#             rolling_stats = tree_map(lambda s0, s1: (1 - learn_rate) * s0 + learn_rate * s1,
#                                      rolling_stats,
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
