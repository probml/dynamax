import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import StandardHMM


class CategoricalRegressionHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 num_classes,
                 feature_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    @property
    def emission_shape(self):
        return ()

    @property
    def covariates_shape(self):
        return (self.feature_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        emission_weights = jr.normal(key1, (self.num_states, self.num_classes, self.feature_dim))
        emission_biases = jr.normal(key2, (self.num_states, self.num_classes))

        params = dict(weights=emission_weights, biases=emission_biases)
        param_props = dict(weights=ParameterProperties(), biases=ParameterProperties())
        return  params, param_props

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_weights=None,
                   emission_biases=None):
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
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_biases (array, optional): manually specified emission biases. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "prior":
            # technically there's no prior, so just sample standard normals
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jr.normal(key1, (self.num_states, self.num_classes, self.feature_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.num_classes))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(weights=default(emission_weights, _emission_weights),
                                   biases=default(emission_biases, _emission_biases))
        props['emissions'] = dict(weights=ParameterProperties(),
                                  biases=ParameterProperties())
        return params, props

    def emission_distribution(self, params, state, covariates=None):
        logits = params['emissions']['weights'][state] @ covariates
        logits += params['emissions']['biases'][state]
        return tfd.Categorical(logits=logits)
