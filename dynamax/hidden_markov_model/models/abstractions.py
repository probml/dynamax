from abc import abstractmethod, ABC
from dynamax.ssm import SSM
from dynamax.types import Scalar
from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.hidden_markov_model.inference import HMMPosterior, HMMPosteriorFiltered
from dynamax.hidden_markov_model.inference import hmm_filter
from dynamax.hidden_markov_model.inference import hmm_posterior_mode
from dynamax.hidden_markov_model.inference import hmm_smoother
from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother
from dynamax.utils.optimize import run_gradient_descent
from dynamax.utils.utils import pytree_slice
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Float, Array, PyTree
import optax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Any, Optional, Tuple
from typing_extensions import Protocol


class HMMParameterSet(Protocol):
    """Container for HMM parameters.

    :param initial: (ParameterSet) initial distribution parameters
    :param transitions: (ParameterSet) transition distribution parameters
    :param emissions: (ParameterSet) emission distribution parameters
    """
    initial: ParameterSet
    transitions: ParameterSet
    emissions: ParameterSet


class HMMPropertySet(Protocol):
    """Container for properties of HMM parameter properties.

    :param initial: (PropertySet) initial distribution properties
    :param transitions: (PropertySet) transition distribution properties
    :param emissions: (PropertySet) emission distribution properties
    """
    initial: PropertySet
    transitions: PropertySet
    emissions: PropertySet



class HMMInitialState(ABC):
    """Abstract class for HMM initial distributions.

    """
    def __init__(self,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50) -> None:
        self.m_step_optimizer = m_step_optimizer
        self.m_step_num_iters = m_step_num_iters

    @abstractmethod
    def distribution(self,
                     params: ParameterSet,
                     inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        """Return a distribution over the initial latent state

        Returns:
            conditional distribution of initial state.

        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self,
                   key: jr.PRNGKey=None,
                   method: str="prior",
                   **kwargs
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator
            method: specifies the type of initialization

        Returns:
            tuple of parameters and their corresponding properties
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the initial distribution parameters.

        Args:
            params: initial distribution parameters

        """
        raise NotImplementedError

    def _compute_initial_probs(self, params, inputs=None):
        return self.initial_distribution(params, inputs).probs_parameter()

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> PyTree:
        """Collect sufficient statistics for updating the initial distribution parameters.

        Args:
            params: initial distribution parameters
            posterior: posterior distribution over latent states
            inputs: optional inputs

        Returns:
            PyTree of sufficient statistics for updating the initial distribution

        """
        return posterior.smoothed_probs[0], pytree_slice(inputs, 0)

    def initialize_m_step_state(self,
                                params: ParameterSet,
                                props: PropertySet):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.

        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self,
               params: ParameterSet,
               props: PropertySet,
               batch_stats: PyTree,
               m_step_state: Any,
               scale: float=1.0
    ) -> ParameterSet:
        """Perform an M-step on the initial distribution parameters.

        Args:
            params: current initial distribution parameters
            props: parameter properties
            batch_stats: PyTree of sufficient statistics from each sequence, as output by :meth:`collect_suff_stats`.
            m_step_state: any state required for the M-step
            scale: how to scale the objective

        Returns:
            Parameters that maximize the expected log joint probability.

        """

        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                expected_initial_state, inpt = stats
                log_initial_prob = jnp.log(self._compute_initial_probs(params, inpt))
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
        params = from_unconstrained(unc_params, props)
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
    def distribution(self,
                     params: ParameterSet,
                     state: int,
                     inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        """Return a distribution over the next latent state

        Args:
            params: transition parameters
            state: current latent state
            inputs: current inputs

        Returns:
            conditional distribution of next state.

        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self,
                   key: jr.PRNGKey=None,
                   method: str="prior",
                   **kwargs
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator
            method: specifies the type of initialization

        Returns:
            tuple of parameters and their corresponding properties

        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the transition distribution parameters.

        Args:
            params: transition distribution parameters

        """
        raise NotImplementedError

    def _compute_transition_matrices(self, params, inputs=None):
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

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> PyTree:
        """Collect sufficient statistics for updating the transition distribution parameters.

        Args:
            params: transition distribution parameters
            posterior: posterior distribution over latent states
            inputs: optional inputs

        Returns:
            PyTree of sufficient statistics for updating the transition distribution

        """
        return posterior.trans_probs, pytree_slice(inputs, slice(1, None))

    def initialize_m_step_state(self, params: ParameterSet, props:PropertySet) -> Any:
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.

        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self,
               params: ParameterSet,
               props: PropertySet,
               batch_stats: PyTree,
               m_step_state: Any,
               scale: float=1.0
    ) -> ParameterSet:
        """Perform an M-step on the transition distribution parameters.

        Args:
            params: current transition distribution parameters
            props: parameter properties
            batch_stats: PyTree of sufficient statistics from each sequence, as output by :meth:`collect_suff_stats`.
            m_step_state: any state required for the M-step
            scale: how to scale the objective

        Returns:
            Parameters that maximize the expected log joint probability.

        """
        unc_params = to_unconstrained(params, props)

        # Minimize the negative expected log joint probability
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                expected_transitions, inputs = stats
                log_trans_matrix = jnp.log(self._compute_transition_matrices(params, inputs))
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
        params = from_unconstrained(unc_params, props)
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
    def emission_shape(self) -> Tuple[int]:
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's emissions.

        For example, a Gaussian HMM with D dimensional emissions would return (D,).
        """
        raise NotImplementedError

    @abstractmethod
    def distribution(self,
                     params: ParameterSet,
                     state: int,
                     inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        """Return a distribution over the emission

        Args:
            params: emission parameters
            state: current latent state
            inputs: current inputs

        Returns:
            conditional distribution of the emission

        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self,
                   key: jr.PRNGKey=None,
                   method: str="prior",
                   **kwargs
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator
            method: specifies the type of initialization

        Returns:
            tuple of parameters and their corresponding properties

        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self, params: ParameterSet) -> Scalar:
        """Compute the log prior probability of the transition distribution parameters.

        Args:
            params: transition distribution parameters

        """
        raise NotImplementedError

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, inpt: \
            vmap(lambda state: self.distribution(params, state, inpt).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, inputs)

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           emissions: Float[Array, "num_timesteps emission_dim"],
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> PyTree:
        """Collect sufficient statistics for updating the emission distribution parameters.

        Args:
            params: emission distribution parameters
            posterior: posterior distribution over latent states
            emissions: observed emissions
            inputs: optional inputs

        Returns:
            PyTree of sufficient statistics for updating the emission distribution

        """
        return posterior.smoothed_probs, emissions, inputs

    def initialize_m_step_state(self, params: ParameterSet, props:PropertySet) -> Any:
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.

        """
        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)
        return self.m_step_optimizer.init(unc_params)

    def m_step(self,
               params: ParameterSet,
               props: PropertySet,
               batch_stats: PyTree,
               m_step_state: Any,
               scale: float=1.0
    ) -> ParameterSet:
        """Perform an M-step on the emission distribution parameters.

        Args:
            params: current emission distribution parameters
            props: parameter properties
            batch_stats: PyTree of sufficient statistics from each sequence, as output by :meth:`collect_suff_stats`.
            m_step_state: any state required for the M-step
            scale: how to scale the objective

        Returns:
            Parameters that maximize the expected log joint probability.

        """

        # Extract the remaining unconstrained params, which should only be for the emissions.
        unc_params = to_unconstrained(params, props)

        # the objective is the negative expected log likelihood (and the log prior of the emission params)
        def neg_expected_log_joint(unc_params):
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                expected_states, emissions, inputs = stats
                log_likelihoods = self._compute_conditional_logliks(params, emissions, inputs)
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
        params = from_unconstrained(unc_params, props)
        return params, m_step_state


class HMM(SSM):
    """Abstract base class of Hidden Markov Models (HMMs).

    The model is defined as follows

    $$z_1 \mid u_1 \sim \mathrm{Cat}(\pi_0(u_1, \\theta_{\mathsf{init}}))$$
    $$z_t \mid z_{t-1}, u_t, \\theta \sim \mathrm{Cat}(\pi(z_{t-1}, u_t, \\theta_{\mathsf{trans}}))$$
    $$y_t | z_t, u_t, \\theta \sim p(y_t \mid z_t, u_t, \\theta_{\mathsf{emis}})$$

    where $z_t \in \{1,\ldots,K\}$ is a *discrete* latent state.
    There are parameters for the initial distribution, the transition distribution,
    and the emission distribution:

    $$\\theta = (\\theta_{\mathsf{init}}, \\theta_{\mathsf{trans}}, \\theta_{\mathsf{emis}})$$

    For "standard" models, we will assume the initial distribution is fixed and the transitions
    follow a simple transition matrix,

    $$z_1 \mid u_1 \sim \mathrm{Cat}(\pi_0)$$
    $$z_t \mid z_{t-1}=k \sim \mathrm{Cat}(\pi_{z_k})$$

    where $\\theta_{\mathsf{init}} = \pi_0$ and $\\theta_{\mathsf{trans}} = \{\pi_k\}_{k=1}^K$.

    The parameters are stored in a :class:`HMMParameterSet` object.

    We have implemented many subclasses of `HMM` for various emission distributions.

    :param num_states: number of discrete states
    :param initial_component: object encapsulating the initial distribution
    :param transition_component: object encapsulating the transition distribution
    :param emission_component: object encapsulating the emission distribution

    """
    def __init__(self,
                 num_states : int,
                 initial_component : HMMInitialState,
                 transition_component : HMMTransitions,
                 emission_component : HMMEmissions):
        self.num_states = num_states
        self.initial_component = initial_component
        self.transition_component = transition_component
        self.emission_component = emission_component

    # Implement the SSM abstract methods by passing on to the components
    @property
    def emission_shape(self):
        return self.emission_component.emission_shape

    def initial_distribution(self, params, inputs=None):
        return self.initial_component.distribution(params.initial, inputs=inputs)

    def transition_distribution(self, params, state, inputs=None):
        return self.transition_component.distribution(params.transitions, state, inputs=inputs)

    def emission_distribution(self, params, state, inputs=None):
        return self.emission_component.distribution(params.emissions, state, inputs=inputs)

    def log_prior(self, params):
        lp = self.initial_component.log_prior(params.initial)
        lp += self.transition_component.log_prior(params.transitions)
        lp += self.emission_component.log_prior(params.emissions)
        return lp

    # The inference functions all need the same arguments
    def _inference_args(self, params, emissions, inputs):
        return (self.initial_component._compute_initial_probs(params.initial, inputs),
                self.transition_component._compute_transition_matrices(params.transitions, inputs),
                self.emission_component._compute_conditional_logliks(params.emissions, emissions, inputs))

    # Convenience wrappers for the inference code
    def marginal_log_prob(self, params, emissions, inputs=None):
        post = hmm_filter(*self._inference_args(params, emissions, inputs))
        return post.marginal_loglik

    def most_likely_states(self, params, emissions, inputs=None):
        return hmm_posterior_mode(*self._inference_args(params, emissions, inputs))

    def filter(self, params, emissions, inputs=None):
        return hmm_filter(*self._inference_args(params, emissions, inputs))

    def smoother(self, params, emissions, inputs=None):
        return hmm_smoother(*self._inference_args(params, emissions, inputs))

    # Expectation-maximization (EM) code
    def e_step(self, params, emissions, inputs=None):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args)

        initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        transition_stats = self.transition_component.collect_suff_stats(params.transitions, posterior, inputs)
        emission_stats = self.emission_component.collect_suff_stats(params.emissions, posterior, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def initialize_m_step_state(self, params, props):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        initial_m_step_state = self.initial_component.initialize_m_step_state(params.initial, props.initial)
        transitions_m_step_state = self.transition_component.initialize_m_step_state(params.transitions, props.transitions)
        emissions_m_step_state = self.emission_component.initialize_m_step_state(params.emissions, props.emissions)
        return initial_m_step_state, transitions_m_step_state, emissions_m_step_state

    def m_step(self, params, props, batch_stats, m_step_state):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_m_step_state = self.initial_component.m_step(params.initial, props.initial, batch_initial_stats, initial_m_step_state)
        transition_params, transitions_m_step_state = self.transition_component.m_step(params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state)
        emission_params, emissions_m_step_state = self.emission_component.m_step(params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state)
        params = params._replace(initial=initial_params, transitions=transition_params, emissions=emission_params)
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
#             expected_log_prob: Mean expected log prob of each epoch.

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
