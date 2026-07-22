"""
Linear regression hidden Markov model (HMM) with state-dependent emission weights and
input-driven initial-state and transition distributions.
"""
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import optax

from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMInitialState, HMMTransitions
from dynamax.hidden_markov_model.models.abstractions import HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionHMMEmissions
from dynamax.hidden_markov_model.models.linreg_hmm import ParamsLinearRegressionHMMEmissions
from dynamax.types import Scalar

from typing import NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float, Int


class ParamsInputDrivenHMMInitialState(NamedTuple):
    """Parameters for the initial distribution of an input-driven HMM."""
    weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states"], ParameterProperties]


class ParamsInputDrivenHMMTransitions(NamedTuple):
    """Parameters for the transitions of an input-driven HMM."""
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]
    biases: Union[Float[Array, "num_states num_states"], ParameterProperties]


class ParamsInputDrivenLinearRegressionHMM(NamedTuple):
    """Parameters for an input-driven linear regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class InputDrivenHMMInitialState(HMMInitialState):
    """
    HMM initial-state distribution for an input-driven HMM.
    The initial-state probabilities depend on the input at the first timestep:
        P(z_1 = k | u_1) = softmax(W_k @ u_1 + b_k)
    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim

    def distribution(self, params: ParamsInputDrivenHMMInitialState, inputs=Float[Array, " input_dim"]) -> tfd.Distribution:
        """Return the distribution object of the initial distribution."""
        logits = params.weights @ inputs + params.biases
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key: Optional[Array] = None,
            method: str = "prior",
            **kwargs
    ) -> Tuple[ParamsInputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState]:
        """Initialize the model parameters and their corresponding properties."""

        if key is None:
            raise ValueError("key must be provided.")

        # Initialize the initial probabilities with small random weights
        key_w, key_b = jr.split(key)
        weights = jr.normal(key_w, (self.num_states, self.input_dim)) * 0.01
        biases = jr.normal(key_b, (self.num_states,)) * 0.01

        # Package the results into dictionaries
        params = ParamsInputDrivenHMMInitialState(weights=weights, biases=biases)
        props = ParamsInputDrivenHMMInitialState(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMInitialState) -> Scalar:
        """Compute the log prior of the parameters."""
        return 0.0


class InputDrivenHMMTransitions(HMMTransitions):
    """
    HMM transitions for an input-driven HMM.
    The transition probabilities depend on external inputs/covariates:
        P(z_t | z_{t-1}, u_t) where u_t are inputs at time t

    For each previous state j, we use multinomial logistic regression:
        P(z_t = k | z_{t-1} = j, u_t) = softmax(W_j @ u_t + b_j)[k]
    """

    def __init__(
            self,
            num_states: int,
            input_dim: int,
            m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
            m_step_num_iters: int = 50
    ):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim

    def distribution(
            self,
            params: ParamsInputDrivenHMMTransitions,
            state: Union[int, Int[Array, ""]],
            inputs: Float[Array, " input_dim"]) -> tfd.Distribution:
        """
        Return the distribution over the next state given the current state and input.
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for input-driven transitions")

        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Categorical(logits=logits)

    def initialize(
            self,
            key: Optional[Array] = None,
            method: str = "prior",
            weights=None,
            biases=None,
            **kwargs
    ) -> Tuple[ParamsInputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions]:
        """Initialize the model parameters and their corresponding properties."""

        if key is None:
            raise ValueError("key must be provided.")

        # Initialize with small random weights so transitions start near uniform
        key_w, key_b = jr.split(key)
        _weights = jr.normal(key_w, (self.num_states, self.num_states, self.input_dim)) * 0.01
        _biases = jr.normal(key_b, (self.num_states, self.num_states)) * 0.01

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsInputDrivenHMMTransitions(
            weights=default(weights, _weights),
            biases=default(biases, _biases))
        props = ParamsInputDrivenHMMTransitions(
            weights=ParameterProperties(),
            biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenHMMTransitions) -> Scalar:
        """Return the log-prior probability of the emission parameters.

        Currently, there is no prior so this function returns 0.
        """
        return 0.0


class InputDrivenLinearRegressionHMM(HMM):
    r"""An HMM with linear-regression emissions whose initial-state and transition
    distributions are driven by an external input, rather than fixed.

    Let $y_t \in \mathbb{R}^N$ and $u_t \in \mathbb{R}^M$ denote vector-valued emissions
    and inputs at time $t$, respectively. The initial distribution (see `InputDrivenHMMInitialState`)
    and the transition distribution (see `InputDrivenHMMTransitions`) are both multinomial logistic
    regressions on the input,

    $$p(z_1 \mid u_1, \theta) = \mathrm{Cat}(z_1 \mid \mathrm{softmax}(W^{\mathsf{init}} u_1 + b^{\mathsf{init}}))$$

    with *initial weights* $W^{\mathsf{init}} \in \mathbb{R}^{K \times M}$ and
    *initial biases* $b^{\mathsf{init}} \in \mathbb{R}^K$, and

    $$p(z_t \mid z_{t-1}, u_t, \theta) = \mathrm{Cat}(z_t \mid \mathrm{softmax}(W^{\mathsf{trans}}_{z_{t-1}} u_t + b^{\mathsf{trans}}_{z_{t-1}}))$$

    with *transition weights* $W_j^{\mathsf{trans}} \in \mathbb{R}^{K \times M}$ and *transition biases* $b_j^{\mathsf{trans}} \in \mathbb{R}^{K}$.

    The emission distribution is a state-dependent linear regression, as in `LinearRegressionHMM`,

    $$p(y_t \mid z_t, u_t, \theta) = \mathcal{N}(y_t \mid W^{\mathsf{emis}}_{z_t} u_t + b^{\mathsf{emis}}_{z_t}, \Sigma_{z_t})$$

    with *emission weights* $W_k \in \mathbb{R}^{N \times M}$, *emission biases* $b_k \in \mathbb{R}^N$,
    and *emission covariances* $\Sigma_k \in \mathbb{R}_{\succeq 0}^{N \times N}$.

    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param emission_dim: emission dimension $N$
    :param m_step_optimizer: ``optax`` optimizer, like Adam, used for the transition M-step.
    :param m_step_num_iters: number of optimizer steps per M-step.
    """

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = InputDrivenHMMInitialState(num_states, input_dim)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = LinearRegressionHMMEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   transition_weights: Optional[Float[Array, "num_states num_states input_dim"]]=None,
                   transition_biases: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Currently only "prior" is supported.
            transition_weights: manually specified transition weights.
            transition_biases: manually specified transition biases.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, weights=transition_weights, biases=transition_biases)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsInputDrivenLinearRegressionHMM(**params), ParamsInputDrivenLinearRegressionHMM(**props)
