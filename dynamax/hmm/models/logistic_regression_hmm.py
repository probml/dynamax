import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import StandardHMM


class LogisticRegressionHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 feature_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_matrices_variance=1e8):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.feature_dim = feature_dim
        self.emission_weights_variance = emission_matrices_variance

    @property
    def emission_shape(self):
        return ()

    @property
    def covariates_shape(self):
        return (self.feature_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        emission_weights = jr.normal(key1, (self.num_states, self.feature_dim))
        emission_biases = jr.normal(key2, (self.num_states,))

        params = dict(weights=emission_weights, biases=emission_biases)
        param_props = dict(weights=ParameterProperties(), biases=ParameterProperties())
        return params, param_props

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Normal(0, self.emission_weights_variance).log_prob(params['emissions']['weights']).sum()
        return lp

    def emission_distribution(self, params, state, covariates):
        logits = params['emissions']['weights'][state] @ covariates
        logits += params['emissions']['biases'][state]
        return tfd.Bernoulli(logits=logits)
