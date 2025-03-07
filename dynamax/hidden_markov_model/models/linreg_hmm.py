"""
Linear regression hidden Markov model (HMM) with state-dependent weights.
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float, Int
from tensorflow_probability.substrates import jax as tfp

from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector

tfd = tfp.distributions
tfb = tfp.bijectors

class ParamsLinearRegressionHMMEmissions(NamedTuple):
    """Parameters for the emissions of a linear regression HMM."""
    weights: Union[Float[Array, "state_dim emission_dim input_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class ParamsLinearRegressionHMM(NamedTuple):
    """Parameters for a linear regression HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class LinearRegressionHMMEmissions(HMMEmissions):
    r"""Emissions for a linear regression hidden Markov model (HMM).

    Args:
        num_states: number of discrete states $K$
        input_dim: input dimension $M$
        emission_dim: emission dimension $N$
    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int):
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim

    @property
    def emission_shape(self) -> Tuple[int]:
        """Return the shape of the emission distribution."""
        return (self.emission_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances: Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions: Optional[Float[Array, "num_timesteps emission_dim"]]=None
                   ) -> Tuple[ParamsLinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions]:
        """Initialize the emission parameters and their corresponding properties.
        
        You can either specify parameters manually via the keyword arguments, or you can have

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.input_dim))
            _emission_biases = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.input_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.emission_dim))
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLinearRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsLinearRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def distribution(
            self,
            params: ParamsLinearRegressionHMMEmissions,
            state: Union[int, Int[Array, ""]],
            inputs: Float[Array, " input_dim"]
    ):
        """Return the emission distribution for a given state."""
        prediction = params.weights[state] @ inputs
        prediction +=  params.biases[state]
        return tfd.MultivariateNormalFullCovariance(prediction, params.covs[state])

    def log_prior(self, params: ParamsLinearRegressionHMMEmissions) -> float:
        """Return the log-prior probability of the emission parameters.
        
        Currently, there is no prior so this function returns 0.
        """
        return 0.0

    # Expectation-maximization (EM) code
    def collect_suff_stats(
            self,
            params: ParamsLinearRegressionHMMEmissions,
            posterior: HMMPosterior,
            emissions: Float[Array, "num_timesteps emission_dim"],
            inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
    ) -> Dict[str, Float[Array, "..."]]:
        """Collect sufficient statistics for the emission parameters."""
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, inputs)
        sum_y = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, inputs)
        sum_xyT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, emissions)
        sum_yyT = jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_y=sum_y, sum_xxT=sum_xxT, sum_xyT=sum_xyT, sum_yyT=sum_yyT)

    def initialize_m_step_state(self, params, props) -> None:
        """Initialize the state for the M-step of the EM algorithm."""
        return None

    def m_step(
            self,
            params: ParamsLinearRegressionHMMEmissions,
            props: ParamsLinearRegressionHMMEmissions,
            batch_stats: Dict[str, Float[Array, "..."]],
            m_step_state: Any
    ) -> Tuple[ParamsLinearRegressionHMMEmissions, Any]:
        """Perform the M-step of the EM algorithm."""
        
        def _single_m_step(stats):
            """Perform the M-step for a single state."""
            sum_w = stats['sum_w']
            sum_x = stats['sum_x']
            sum_y = stats['sum_y']
            sum_xxT = stats['sum_xxT']
            sum_xyT = stats['sum_xyT']
            sum_yyT = stats['sum_yyT']

            # Make block matrices for stacking features (x) and bias (1)
            sum_x1x1T = jnp.block(
                [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
                 [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
            )
            sum_x1yT = jnp.vstack([sum_xyT, sum_y])

            # Solve for the optimal A, b, and Sigma
            Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
            Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
            Sigma = 0.5 * (Sigma + Sigma.T)                 # for numerical stability
            return Ab[:, :-1], Ab[:, -1], Sigma

        emission_stats = pytree_sum(batch_stats, axis=0)
        As, bs, Sigmas = vmap(_single_m_step)(emission_stats)
        params = params._replace(weights=As, biases=bs, covs=Sigmas)
        return params, m_step_state


class LinearRegressionHMM(HMM):
    r"""An HMM whose emissions come from a linear regression with state-dependent weights.
    This is also known as a *switching linear regression* model.

    Let $y_t \in \mathbb{R}^N$ and $u_t \in \mathbb{R}^M$ denote vector-valued emissions
    and inputs at time $t$, respectively. In this model, the emission distribution is,

    $$p(y_t \mid z_t, u_t, \theta) = \mathcal{N}(y_{t} \mid W_{z_t} u_t + b_{z_t}, \Sigma_{z_t})$$

    with *emission weights* $W_k \in \mathbb{R}^{N \times M}$, *emission biases* $b_k \in \mathbb{R}^N$,
    and *emission covariances* $\Sigma_k \in \mathbb{R}_{\succeq 0}^{N \times N}$.

    The emissions parameters are $\theta = \{W_k, b_k, \Sigma_k\}_{k=1}^K$.

    We do not place a prior on the emission parameters.

    *Note: in the future we add a* matrix-normal-inverse-Wishart_ *prior (see pg 576).*

    .. _matrix-normal-inverse-Wishart: https://github.com/probml/pml2-book

    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param emission_dim: emission dimension $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.

    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LinearRegressionHMMEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
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
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsLinearRegressionHMM(**params), ParamsLinearRegressionHMM(**props)
