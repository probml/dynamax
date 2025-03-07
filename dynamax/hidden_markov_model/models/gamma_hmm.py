"""Hidden Markov Model with Gamma emissions."""
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import Scalar
import optax
from typing import NamedTuple, Optional, Tuple, Union


class ParamsGammaHMMEmissions(NamedTuple):
    """Parameters for the Gamma emissions of an HMM."""
    concentration: Union[Float[Array, " state_dim"], ParameterProperties]
    rate: Union[Float[Array, " state_dim"], ParameterProperties]


class GammaHMMEmissions(HMMEmissions):
    r"""Gamma emissions for an HMM.
    
    :param num_states: number of discrete states $K$
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.
    """
    def __init__(
        self,
        num_states: int,
        m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
        m_step_num_iters: int = 50,
    ):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states

    @property
    def emission_shape(self) -> Tuple:
        """Shape of the emission distribution."""
        return ()

    def initialize(
        self,
        key: Array = jr.PRNGKey(0),
        method="prior",
        emission_concentrations: Optional[Float[Array, " num_states"]] = None,
        emission_rates: Optional[Float[Array, " num_states"]] = None,
        emissions: Optional[Float[Array, " num_timesteps"]] = None,
        # ) -> Tuple[ParamsGammaHMMEmissions, ParamsGammaHMMEmissions]:
    ) -> Tuple[ParamsGammaHMMEmissions, ParamsGammaHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.
        
        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_concentrations: manually specified emission concentrations.
            emission_rates: manually specified emission rates.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans

            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, 1))

            _emission_concentrations = jnp.ones((self.num_states,))
            _emission_rates = jnp.ravel(1.0 / km.cluster_centers_)

        elif method.lower() == "prior":
            _emission_concentrations = jnp.ones((self.num_states,))
            _emission_rates = jr.exponential(key, (self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsGammaHMMEmissions(
            concentration=default(emission_concentrations, _emission_concentrations),
            rate=default(emission_rates, _emission_rates),
        )
        props = ParamsGammaHMMEmissions(
            concentration=ParameterProperties(constrainer=tfb.Softplus()),
            rate=ParameterProperties(constrainer=tfb.Softplus()),
        )
        return params, props

    def log_prior(self, params) -> float:
        """Compute the log-prior probability of the parameters."""
        return 0.0

    def distribution(self, params: ParamsGammaHMMEmissions, state, inputs=None) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        return tfd.Gamma(concentration=params.concentration[state], rate=params.rate[state])


class ParamsGammaHMM(NamedTuple):
    """Parameters for a Gamma HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsGammaHMMEmissions


class GammaHMM(HMM):
    r"""An HMM whose emissions come from a gamma distribution.

    Let $y_t \in \mathbb{R}_+$ denote non-negative emissions.
    In this model, the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \mathrm{Ga}(y_{t} \mid \alpha_{z_t}, \beta_{z_t})$$

    with *emission concentration* $\alpha_k \in \mathbb{R}_+$ and *emission rate* $\beta_k \in \mathbb{R}_+$.


    :param num_states: number of discrete states $K$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """

    def __init__(
        self,
        num_states: int,
        initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
        transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
        transition_matrix_stickiness: Scalar = 0.0,
        m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
        m_step_num_iters: int = 50,
    ):
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(
            num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness
        )
        emission_component = GammaHMMEmissions(
            num_states, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters
        )
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(
        self,
        key: Array = jr.PRNGKey(0),
        method: str = "prior",
        initial_probs: Optional[Float[Array, " num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
        emission_concentrations: Optional[Float[Array, " num_states"]] = None,
        emission_rates: Optional[Float[Array, " num_states"]] = None,
        emissions: Optional[Float[Array, " num_timesteps"]] = None,
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
            emission_concentrations: manually specified emission concentrations.
            emission_rates: manually specified emission rates.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        params["transitions"], props["transitions"] = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix
        )
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3,
            method=method,
            emission_concentrations=emission_concentrations,
            emission_rates=emission_rates,
            emissions=emissions,
        )
        return ParamsGammaHMM(**params), ParamsGammaHMM(**props)
