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

    def emission_distribution(self, params, state, covariates=None):
        logits = params['emissions']['weights'][state] @ covariates
        logits += params['emissions']['biases'][state]
        return tfd.Categorical(logits=logits)
