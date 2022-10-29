import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import ExponentialFamilyHMM


class PoissonHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1,
                 emission_prior_rate=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_concentration = emission_prior_concentration
        self.emission_prior_rate = emission_prior_rate

    @property
    def emission_shape(self):
        return (self.emission_dim,)

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
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

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
        params['emissions'] = dict(rates=emission_rates)
        props['emissions'] = dict(rates=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.Independent(tfd.Poisson(rate=params['emissions']['rates'][state]),
                               reinterpreted_batch_ndims=1)

    def _zeros_like_suff_stats(self):
        """Return dataclass containing 'event_shape' of each sufficient statistic."""
        return dict(
            sum_w=jnp.zeros((self.num_states, 1)),
            sum_x=jnp.zeros((self.num_states, self.emission_dim)),
        )

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Gamma(self.emission_prior_concentration, self.emission_prior_rate).log_prob(
            params['emissions']['rates']).sum()
        return lp

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_w = jnp.einsum("tk->k", expected_states)[:, None]
        sum_x = jnp.einsum("tk, ti->ki", expected_states, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x)

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['rates'].trainable:
            post_concentration = self.emission_prior_concentration + emission_stats['sum_x']
            post_rate = self.emission_prior_rate + emission_stats['sum_w']
            params['emissions']['rates'] = tfd.Gamma(post_concentration, post_rate).mode()
        return params
