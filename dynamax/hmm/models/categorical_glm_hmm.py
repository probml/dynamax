import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.abstractions import HMM, HMMEmissions
from dynamax.hmm.models.initial import StandardHMMInitialState
from dynamax.hmm.models.transitions import StandardHMMTransitions


class CategoricalRegressionHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 num_classes,
                 covariate_dim):
        """_summary_

        Args:
            emission_probs (_type_): _description_
        """
        self.num_states = num_states
        self.num_classes = num_classes
        self.feature_dim = covariate_dim

    @property
    def emission_shape(self):
        return ()

    @property
    def covariates_shape(self):
        return (self.feature_dim,)

    def initialize(self, key=jr.PRNGKey(0), method="prior", emission_weights=None, emission_biases=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_biases (array, optional): manually specified emission biases. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "prior":
            # technically there's no prior, so just sample standard normals
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jr.normal(key1, (self.num_states, self.num_classes, self.feature_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.num_classes))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = dict(weights=default(emission_weights, _emission_weights),
                      biases=default(emission_biases, _emission_biases))
        props = dict(weights=ParameterProperties(), biases=ParameterProperties())
        return params, props

    def distribution(self, params, state, covariates=None):
        logits = params['weights'][state] @ covariates
        logits += params['biases'][state]
        return tfd.Categorical(logits=logits)


class CategoricalRegressionHMM(HMM):
    def __init__(self, num_states: int,
                 num_classes: int,
                 covariate_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = CategoricalRegressionHMMEmissions(num_states, num_classes, covariate_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=None,
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_weights: jnp.array=None,
                   emission_biases: jnp.array=None):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key=key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases)
        return params, props
