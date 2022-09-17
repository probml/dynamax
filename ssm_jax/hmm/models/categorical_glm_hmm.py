import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from ssm_jax.parameters import ParameterProperties
from ssm_jax.hmm.models.base import StandardHMM


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

    def random_initialization(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_weights = jr.normal(key3, (self.num_states, self.num_classes, self.feature_dim))
        emission_biases = jr.normal(key4, (self.num_states, self.num_classes))

        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(weights=emission_weights, biases=emission_biases))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(weights=ParameterProperties(), biases=ParameterProperties()))
        return  params, param_props

    def emission_distribution(self, params, state, **covariates):
        logits = params['emissions']['weights'][state] @ covariates['features']
        logits += params['emissions']['biases'][state]
        return tfd.Categorical(logits=logits)
