import jax.numpy as jnp
import jax.random as jr
from jax.nn import one_hot
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.abstractions import HMM, HMMEmissions
from dynamax.hmm.models.initial import StandardHMMInitialState
from dynamax.hmm.models.transitions import StandardHMMTransitions
from dynamax.utils import pytree_sum


class CategoricalHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_classes,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            emission_probs (_type_): _description_
        """
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.emission_prior_concentration = emission_prior_concentration  * jnp.ones(num_classes)

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, covariates=None):
        return tfd.Independent(
            tfd.Categorical(probs=params['probs'][state]),
            reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params['probs']).sum()

    def initialize(self, key=jr.PRNGKey(0), method="prior", emission_probs=None):
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
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
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
        params = dict(probs=emission_probs)
        props = dict(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, covariates=None):
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions, self.num_classes)
        return dict(sum_x=jnp.einsum("tk,tdi->kdi", expected_states, x))

    def m_step(self, params, props, batch_stats):
        if props['probs'].trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            params['probs'] = tfd.Dirichlet(
                self.emission_prior_concentration + emission_stats['sum_x']).mode()
        return params


class CategoricalHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = CategoricalHMMEmissions(num_states, emission_dim, num_classes, emission_prior_concentration=emission_prior_concentration)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=None,
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_probs: jnp.array=None):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_probs=emission_probs)
        return params, props
