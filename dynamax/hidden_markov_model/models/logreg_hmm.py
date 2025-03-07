"""
Logistic Regression Hidden Markov Model
"""
from typing import NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import IntScalar, Scalar


class ParamsLogisticRegressionHMMEmissions(NamedTuple):
    """Parameters for the logistic regression emissions in an HMM."""
    weights: Union[Float[Array, "state_dim input_dim"], ParameterProperties]
    biases: Union[Float[Array, " state_dim"], ParameterProperties]


class LogisticRegressionHMMEmissions(HMMEmissions):
    r"""Logistic regression emissions for an HMM.
    
    Args:
        num_states: number of discrete states $K$
        input_dim: input dimension $M$
        emission_matrices_scale: $\varsigma$
        m_step_optimizer: ``optax`` optimizer, like Adam.
        m_step_num_iters: number of optimizer steps per M-step.
    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_matrices_scale: Scalar = 1e8,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_weights_scale = emission_matrices_scale

    @property
    def emission_shape(self) -> Tuple:
        """Shape of the emission distribution."""
        return ()

    @property
    def inputs_shape(self) -> Tuple[int]:
        """Shape of the inputs to the emission distribution."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array = jr.PRNGKey(0),
                   method: str = "prior",
                   emission_weights: Optional[Float[Array, "num_states input_dim"]] = None,
                   emission_biases: Optional[Float[Array, " num_states"]] = None,
                   emissions: Optional[Float[Array, " num_timesteps"]] = None,
                   inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
                   ) -> Tuple[ParamsLogisticRegressionHMMEmissions, ParamsLogisticRegressionHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emissions: emissions for initializing the parameters with kmeans.
            inputs: inputs for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            assert inputs is not None, "Need inputs to initialize the model with K-Means!"
            from sklearn.cluster import KMeans

            flat_emissions = emissions.reshape(-1,)
            flat_inputs = inputs.reshape(-1, self.input_dim)
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(flat_inputs)
            _emission_weights = jnp.zeros((self.num_states, self.input_dim))
            _emission_biases = jnp.array([tfb.Sigmoid().inverse(flat_emissions[km.labels_ == k].mean())
                                          for k in range(self.num_states)])

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.input_dim))
            _emission_biases = jr.normal(key2, (self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLogisticRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases))
        props = ParamsLogisticRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsLogisticRegressionHMMEmissions) -> Float[Array, ""]:
        """Log prior probability of the emission parameters."""
        return tfd.Normal(0, self.emission_weights_scale).log_prob(params.weights).sum()

    def distribution(
            self,
            params: ParamsLogisticRegressionHMMEmissions,
            state: IntScalar,
            inputs: Float[Array, "input_dim"]
            ) -> tfd.Distribution:
        """Emission distribution for a given state."""
        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Bernoulli(logits=logits)


class ParamsLogisticRegressionHMM(NamedTuple):
    """Parameters for a logistic regression HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLogisticRegressionHMMEmissions


class LogisticRegressionHMM(HMM):
    r"""An HMM whose emissions come from a logistic regression with state-dependent weights.
    This is also known as a *switching logistic regression* model.

    Let $y_t \in \{0,1\}$ and $u_t \in \mathbb{R}^M$ denote binary emissions
    and inputs at time $t$, respectively. In this model, the emission distribution is,

    $$p(y_t \mid z_t, u_t, \theta) = \mathrm{Bern}(y_{t} \mid \sigma(w_{z_t}^\top u_t + b_{z_t}))$$

    with *emission weights* $w_k \in \mathbb{R}^{M}$ and *emission biases* $b_k \in \mathbb{R}$.

    We use $L_2$ regularization on the emission weights, which can be thought of as a
    Gaussian prior,

    $$p(\theta) \propto \prod_{k=1}^K \prod_{m=1}^M \mathcal{N}(w_{k,m} \mid 0, \varsigma^2)$$

    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_matrices_scale: $\varsigma$
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_matrices_scale: Scalar=1e8,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.inputs_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LogisticRegressionHMMEmissions(num_states, input_dim, emission_matrices_scale=emission_matrices_scale, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self) -> Tuple[int, ...]:
        """Shape of the inputs to the emission distribution."""
        return (self.inputs_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states input_dim"]]=None,
                   emission_biases: Optional[Float[Array, " num_states"]]=None,
                   emissions: Optional[Float[Array, " num_timesteps"]]=None,
                   inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emissions: emissions for initializing the parameters with kmeans.
            inputs: inputs for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emissions=emissions, inputs=inputs)
        return ParamsLogisticRegressionHMM(**params), ParamsLogisticRegressionHMM(**props)
