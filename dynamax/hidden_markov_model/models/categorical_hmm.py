"""Categorical Hidden Markov Model."""
from typing import NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.nn import one_hot
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions
from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import IntScalar, Scalar
from dynamax.utils.utils import pytree_sum


class ParamsCategoricalHMMEmissions(NamedTuple):
    """Parameters for the CategoricalHMM emission distribution."""
    probs: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class ParamsCategoricalHMM(NamedTuple):
    """Parameters for the CategoricalHMM model."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalHMMEmissions


class CategoricalHMMEmissions(HMMEmissions):
    r"""Categorical emissions for a hidden Markov model."""

    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 emission_prior_concentration: Union[Scalar, Float[Array, " num_classes"]]=1.1):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.emission_prior_concentration = emission_prior_concentration  * jnp.ones(num_classes)

    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of the emission distribution."""
        return (self.emission_dim,)

    def distribution(self, params: ParamsCategoricalHMMEmissions, state: IntScalar, inputs=None) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        return tfd.Independent(
            tfd.Categorical(probs=params.probs[state]),
            reinterpreted_batch_ndims=1)

    def log_prior(self, params: ParamsCategoricalHMMEmissions) -> Scalar:
        """Return the log prior probability of the emission parameters."""
        return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params.probs).sum()

    def initialize(self,
                   key:Optional[Array]=jr.PRNGKey(0),
                   method="prior",
                   emission_probs:Optional[Float[Array, "num_states emission_dim num_classes"]]=None
                   ) -> Tuple[ParamsCategoricalHMMEmissions, ParamsCategoricalHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("key must not be None when emission_probs is None")
                prior = tfd.Dirichlet(self.emission_prior_concentration)
                emission_probs = prior.sample(seed=key, sample_shape=(self.num_states, self.emission_dim))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.emission_dim, self.num_classes)
            assert jnp.all(emission_probs >= 0)
            assert jnp.allclose(emission_probs.sum(axis=2), 1.0)

        # Add parameters to the dictionary
        params = ParamsCategoricalHMMEmissions(probs=emission_probs)
        props = ParamsCategoricalHMMEmissions(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        """Collect sufficient statistics for the emission distribution."""
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def initialize_m_step_state(self, params, props):
        """Initialize the m-step state."""
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        """Perform the m-step for the emission distribution."""
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            probs = tfd.Dirichlet(self.emission_prior_concentration + emission_stats['sum_x']).mode()
            params = params._replace(probs=probs)
        return params, m_step_state


class CategoricalHMM(HMM):
    r"""An HMM with conditionally independent categorical emissions.

    Let $y_t \in \{1,\ldots,C\}^N$ denote a vector of $N$ conditionally independent
    categorical emissions from $C$ classes at time $t$. In this model,the emission
    distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathrm{Cat}(y_{tn} \mid \theta_{z_t,n})$$
    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathrm{Dir}(\theta_{k,n}; \gamma 1_C)$$

    with $\theta_{k,n} \in \Delta_C$ for $k=1,\ldots,K$ and $n=1,\ldots,N$ are the
    *emission probabilities* and $\gamma$ is their prior concentration.

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param num_classes: number of multinomial classes $C$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_concentration: $\gamma$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_concentration=1.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = CategoricalHMMEmissions(num_states, emission_dim, num_classes, emission_prior_concentration=emission_prior_concentration)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]]=None
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_probs=emission_probs)
        return ParamsCategoricalHMM(**params), ParamsCategoricalHMM(**props)
