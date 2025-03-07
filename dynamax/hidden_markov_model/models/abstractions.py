"""
This module contains the abstract classes for the components of a Hidden Markov Model (HMM).
"""
from abc import abstractmethod, ABC
from typing import Any, Optional, Tuple, runtime_checkable, Union 
from typing_extensions import Protocol
from dynamax.ssm import SSM
from dynamax.types import IntScalar, Scalar
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
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import Float, Array, PyTree, Real
import optax
from tensorflow_probability.substrates.jax import distributions as tfd


@runtime_checkable
class HMMParameterSet(Protocol):
    """Container for HMM parameters.

    :param initial: (ParameterSet) initial distribution parameters
    :param transitions: (ParameterSet) transition distribution parameters
    :param emissions: (ParameterSet) emission distribution parameters
    """
    @property
    def initial(self) -> ParameterSet:
        """Initial distribution parameters."""
        pass

    @property
    def transitions(self) -> ParameterSet:
        """Transition distribution parameters."""
        pass

    @property
    def emissions(self) -> ParameterSet:
        """Emission distribution parameters."""
        pass


@runtime_checkable
class HMMPropertySet(Protocol):
    """Container for properties of HMM parameter properties.

    :param initial: (PropertySet) initial distribution properties
    :param transitions: (PropertySet) transition distribution properties
    :param emissions: (PropertySet) emission distribution properties
    """
    @property
    def initial(self) -> PropertySet:
        """Initial distribution properties."""
        pass

    @property
    def transitions(self) -> PropertySet:
        """Transition distribution properties."""
        pass

    @property
    def emissions(self) -> PropertySet:
        """Emission distribution properties."""
        pass


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
                     inputs: Optional[Float[Array, " input_dim"]]=None
    ) -> tfd.Distribution:
        """Return a distribution over the initial latent state

        Returns:
            conditional distribution of initial state.

        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self,
                   key: Optional[Array]=None,
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

    def _compute_initial_probs(self, params, inputs:Optional[Array] = None):
        """Compute the initial probabilities for each state."""
        return self.distribution(params, inputs).probs_parameter()

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
                           ) -> Tuple[Float[Array, " num_states"], Optional[Float[Array, " input_dim"]]]:
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
    ) -> Tuple[ParameterSet, Any]:
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
            """Compute the negative expected log joint probability."""
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                """Compute the expected log likelihood for a single sequence."""
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
                     state: IntScalar,
                     inputs: Optional[Float[Array, " input_dim"]]=None
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
                   key: Optional[Array]=None,
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

    def _compute_transition_matrices(self, params, inputs:Optional[Array] = None):
        """Compute the transition matrix for each time step."""
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
                           ) -> Tuple[Float[Array, "..."], Optional[Float[Array, "num_timesteps-1 input_dim"]]]:
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
    ) -> Tuple[ParameterSet, Any]:
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
            """Compute the negative expected log joint probability."""
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                """Compute the expected log likelihood for a single sequence."""
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
                     state: IntScalar,
                     inputs: Optional[Float[Array, " input_dim"]]=None
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
                   key: Optional[Array]=None,
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

    def _compute_conditional_logliks(self, params, emissions, inputs:Optional[Array] = None):
        """Compute the log likelihood of the emissions given the latent states."""
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission, inpt: \
            vmap(lambda state: self.distribution(params, state, inpt).log_prob(emission))(
                jnp.arange(self.num_states))
        return vmap(f)(emissions, inputs)

    def collect_suff_stats(self,
                           params: ParameterSet,
                           posterior: HMMPosterior,
                           emissions: Union[Real[Array, "num_timesteps emission_dim"],
                                            Real[Array, " num_timesteps"]],
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps num_states"],
               Union[Real[Array, "num_timesteps emission_dim"], Real[Array, " num_timesteps"]],
               Optional[Float[Array, "num_timesteps input_dim"]]]:
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
    ) -> Tuple[ParameterSet, Any]:
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
            """Compute the negative expected log joint probability."""
            params = from_unconstrained(unc_params, props)

            def _single_expected_log_like(stats):
                """Compute the expected log likelihood for a single sequence."""
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
    r"""Abstract base class of Hidden Markov Models (HMMs).

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
        """Return the shape of the emission distribution."""
        return self.emission_component.emission_shape

    def initial_distribution(self, params: HMMParameterSet, 
                             inputs:Optional[Array] = None) \
                             -> tfd.Distribution:
        """Return the initial distribution."""
        return self.initial_component.distribution(params.initial, inputs=inputs)

    def transition_distribution(self, params: HMMParameterSet, 
                                state: IntScalar, 
                                inputs:Optional[Array] = None) \
                                -> tfd.Distribution:
        """Return the transition distribution."""
        return self.transition_component.distribution(params.transitions, state, inputs=inputs)

    def emission_distribution(self, params: HMMParameterSet, 
                              state: IntScalar, 
                              inputs:Optional[Array] = None) \
                              -> tfd.Distribution:
        """Return the emission distribution."""
        return self.emission_component.distribution(params.emissions, state, inputs=inputs)

    def log_prior(self, params: HMMParameterSet) -> Scalar:
        """Compute the log prior probability of the model parameters."""
        lp = self.initial_component.log_prior(params.initial)
        lp += self.transition_component.log_prior(params.transitions)
        lp += self.emission_component.log_prior(params.emissions)
        return lp

    # The inference functions all need the same arguments
    def _inference_args(self, params: HMMParameterSet, 
                        emissions: Array, 
                        inputs: Optional[Array]) -> Tuple:
        """Return the arguments needed for inference."""
        return (self.initial_component._compute_initial_probs(params.initial, inputs),
                self.transition_component._compute_transition_matrices(params.transitions, inputs),
                self.emission_component._compute_conditional_logliks(params.emissions, emissions, inputs))

    # Convenience wrappers for the inference code
    def marginal_log_prob(self, params: HMMParameterSet, 
                          emissions: Array, 
                          inputs: Optional[Array]=None) -> float:
        """Compute the marginal log probability of the emissions."""
        post = hmm_filter(*self._inference_args(params, emissions, inputs))
        return post.marginal_loglik

    def most_likely_states(self, params: HMMParameterSet, 
                           emissions: Array, 
                           inputs: Optional[Array]=None) \
                           -> Float[Array, "num_timesteps"]:
        """Compute the most likely sequence of states."""
        return hmm_posterior_mode(*self._inference_args(params, emissions, inputs))

    def filter(self, params: HMMParameterSet, 
               emissions: Array, 
               inputs: Optional[Array]=None) \
               -> HMMPosteriorFiltered:
        """Compute the filtering distributions."""
        return hmm_filter(*self._inference_args(params, emissions, inputs))

    def smoother(self, params: HMMParameterSet, 
                 emissions: Array, 
                 inputs: Optional[Array]=None) \
                 -> HMMPosterior:
        """Compute the posterior smoothing distributions."""
        return hmm_smoother(*self._inference_args(params, emissions, inputs))

    # Expectation-maximization (EM) code
    def e_step(
            self,
            params: HMMParameterSet,
            emissions: Array,
           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
           ) -> Tuple[PyTree, Scalar]:
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args)

        initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        transition_stats = self.transition_component.collect_suff_stats(params.transitions, posterior, inputs)
        emission_stats = self.emission_component.collect_suff_stats(params.emissions, posterior, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def initialize_m_step_state(self, params: HMMParameterSet, props: HMMPropertySet):
        """Initialize any required state for the M step.

        For example, this might include the optimizer state for Adam.
        """
        initial_m_step_state = self.initial_component.initialize_m_step_state(params.initial, props.initial)
        transitions_m_step_state = self.transition_component.initialize_m_step_state(params.transitions, props.transitions)
        emissions_m_step_state = self.emission_component.initialize_m_step_state(params.emissions, props.emissions)
        return initial_m_step_state, transitions_m_step_state, emissions_m_step_state

    def m_step(
            self,
            params: HMMParameterSet,
            props: HMMPropertySet,
            batch_stats: PyTree,
            m_step_state: Any
            ) -> Tuple[HMMParameterSet, Any]:
        """
        Perform an M-step on the model parameters.
        """
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_m_step_state = self.initial_component.m_step(params.initial, props.initial, batch_initial_stats, initial_m_step_state)
        transition_params, transitions_m_step_state = self.transition_component.m_step(params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state)
        emission_params, emissions_m_step_state = self.emission_component.m_step(params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state)
        params = params._replace(initial=initial_params, transitions=transition_params, emissions=emission_params)
        m_step_state = initial_m_step_state, transitions_m_step_state, emissions_m_step_state
        return params, m_step_state
