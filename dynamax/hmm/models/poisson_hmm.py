import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.abstractions import HMM, HMMEmissions
from dynamax.hmm.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hmm.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.utils.utils import pytree_sum
from typing import NamedTuple, Union


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

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1,
                 emission_prior_rate=0.1):
        """_summary_

        Args:
            num_states (_type_): _description_
            emission_dim (_type_): _description_
            initial_probs_concentration (float, optional): _description_. Defaults to 1.1.
            transition_matrix_concentration (float, optional): _description_. Defaults to 1.1.
            emission_prior_concentration (float, optional): _description_. Defaults to 1.1.
            emission_prior_rate (float, optional): _description_. Defaults to 0.1.
        """
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = PoissonHMMEmissions(num_states, emission_dim, emission_prior_concentration=emission_prior_concentration, emission_prior_rate=emission_prior_rate)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_rates=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_rates (array, optional): manually specified emission rates. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_rates=emission_rates)
        return ParamsPoissonHMM(**params), ParamsPoissonHMM(**props)
