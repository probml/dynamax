from typing import NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions
from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum


class ParamsPoissonHMMEmissions(NamedTuple):
    rates: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class PoissonHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_concentration=1.1,
                 emission_prior_rate=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_concentration = emission_prior_concentration
        self.emission_prior_rate = emission_prior_rate

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   emission_rates=None):
        # Initialize the emission probabilities
        if emission_rates is None:
            if method.lower() == "prior":
                prior = tfd.Gamma(self.emission_prior_concentration, self.emission_prior_rate)
                emission_rates = prior.sample(seed=key, sample_shape=(self.num_states, self.emission_dim))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_rates.shape == (self.num_states, self.emission_dim)
            assert jnp.all(emission_rates >= 0)

        # Add parameters to the dictionary
        params = ParamsPoissonHMMEmissions(rates=emission_rates)
        props = ParamsPoissonHMMEmissions(rates=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, params, state, inputs=None):
        return tfd.Independent(tfd.Poisson(rate=params.rates[state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        prior = tfd.Gamma(self.emission_prior_concentration, self.emission_prior_rate)
        return prior.log_prob(params.rates).sum()

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)[:, None]
        sum_x = jnp.einsum("tk, ti->ki", expected_states, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x)

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.rates.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            post_concentration = self.emission_prior_concentration + emission_stats['sum_x']
            post_rate = self.emission_prior_rate + emission_stats['sum_w']
            rates = tfd.Gamma(post_concentration, post_rate).mode()
            params = params._replace(rates=rates)
        return params, m_step_state


class ParamsPoissonHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsPoissonHMMEmissions


class PoissonHMM(HMM):
    r"""An HMM with conditionally independent Poisson emissions.

    Let $y_t \in \{0,1\}^N$ denote a vector of count emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathrm{Po}(y_{tn} \mid \theta_{z_t,n})$$
    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathrm{Ga}(\theta_{k,n}; \gamma_0, \gamma_1)$$

    with $\theta_{k,n} \in \mathbb{R}_+$ for $k=1,\ldots,K$ and $n=1,\ldots,N$ are the
    *emission rates* and $\gamma_0, \gamma_1$ are their prior concentration and rate, respectively.

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_concentration: $\gamma_0$
    :param emission_prior_rate: $\gamma_1$

    """
    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_concentration: Scalar=1.1,
                 emission_prior_rate: Scalar=0.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = PoissonHMMEmissions(num_states, emission_dim, emission_prior_concentration=emission_prior_concentration, emission_prior_rate=emission_prior_rate)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_rates: Optional[Float[Array, "num_states emission_dim"]]=None
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method: method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs: manually specified initial state probabilities. Defaults to None.
            transition_matrix: manually specified transition matrix. Defaults to None.
            emission_rates: manually specified emission probabilities. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_rates=emission_rates)
        return ParamsPoissonHMM(**params), ParamsPoissonHMM(**props)
