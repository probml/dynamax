import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


class BernoulliHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration1=1.1,
                 emission_prior_concentration0=1.1):
        """_summary_
        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_dim
        self.emission_prior_concentration0 = emission_prior_concentration0
        self.emission_prior_concentration1 = emission_prior_concentration1

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def emission_distribution(self, params, state, covariates=None):
        # This model assumes the emissions are a vector of conditionally independent
        # Bernoulli observations. The `reinterpreted_batch_ndims` argument tells
        # `tfd.Independent` that only the last dimension should be considered a "batch"
        # of conditionally independent observations.
        return tfd.Independent(tfd.Bernoulli(probs=params['emissions']['probs'][state]),
                               reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Beta(self.emission_prior_concentration1, self.emission_prior_concentration0).log_prob(
            params['emissions']['probs']).sum()
        return lp

    def initialize(self, key=jr.PRNGKey(0), method="prior", initial_probs=None, transition_matrix=None, emission_probs=None):
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
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
                prior = tfd.Beta(self.emission_prior_concentration1, self.emission_prior_concentration0)
                emission_probs = prior.sample(seed=key, sample_shape=(self.num_states, self.emission_dim))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.emission_dim)
            assert jnp.all(emission_probs >= 0)
            assert jnp.all(emission_probs <= 0)

        # Add parameters to the dictionary
        params['emissions'] = dict(probs=emission_probs)
        props['emissions'] = dict(probs=ParameterProperties(constrainer=tfb.Sigmoid()))
        return params, props

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        sum_x = jnp.zeros((self.num_states, self.emission_dim)),
        sum_1mx = jnp.zeros((self.num_states, self.emission_dim)),
        return (sum_x, sum_1mx)

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_x = jnp.einsum("tk, ti->ki", expected_states, jnp.where(jnp.isnan(emissions), 0, emissions))
        sum_1mx = jnp.einsum("tk, ti->ki", expected_states,
                                jnp.where(jnp.isnan(emissions), 0, 1 - emissions))
        return (sum_x, sum_1mx)

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['probs'].trainable:
            sum_x, sum_1mx = emission_stats
            params['emissions']['probs'] = tfd.Beta(
                self.emission_prior_concentration1 + sum_x,
                self.emission_prior_concentration0 + sum_1mx).mode()
        return params
