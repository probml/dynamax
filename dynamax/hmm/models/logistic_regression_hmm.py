import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import StandardHMM


class LogisticRegressionHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 covariate_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_matrices_scale=1e8):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.covariate_dim = covariate_dim
        self.emission_weights_scale = emission_matrices_scale

    @property
    def emission_shape(self):
        return ()

    @property
    def covariates_shape(self):
        return (self.covariate_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_weights=None,
                   emission_biases=None,
                   emissions=None,
                   covariates=None):
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
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_biases (array, optional): manually specified emission biases. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.
            covariates (array, optional): covariates for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            assert covariates is not None, "Need covariates to initialize the model with K-Means!"
            from sklearn.cluster import KMeans

            flat_emissions = emissions.reshape(-1,)
            flat_covariates = covariates.reshape(-1, self.covariate_dim)
            km = KMeans(self.num_states).fit(flat_covariates)

            _emission_weights = jnp.zeros((self.num_states, self.covariate_dim))
            _emission_biases = jnp.array([tfb.Sigmoid().inverse(flat_emissions[km.labels_ == k].mean())
                                          for k in range(self.num_states)])

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.covariate_dim))
            _emission_biases = jr.normal(key2, (self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(weights=default(emission_weights, _emission_weights),
                                   biases=default(emission_biases, _emission_biases))
        props['emissions'] = dict(weights=ParameterProperties(),
                                  biases=ParameterProperties())
        return params, props

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Normal(0, self.emission_weights_scale).log_prob(params['emissions']['weights']).sum()
        return lp

    def emission_distribution(self, params, state, covariates):
        logits = params['emissions']['weights'][state] @ covariates
        logits += params['emissions']['biases'][state]
        return tfd.Bernoulli(logits=logits)
