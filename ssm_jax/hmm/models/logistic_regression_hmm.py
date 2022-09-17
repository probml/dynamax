import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from ssm_jax.parameters import ParameterProperties
from ssm_jax.hmm.models.base import StandardHMM


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

    def random_initialization(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_weights = jr.normal(key3, (self.num_states, self.feature_dim))
        emission_biases = jr.normal(key4, (self.num_states,))

        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(weights=emission_weights, biases=emission_biases))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(weights=ParameterProperties(), biases=ParameterProperties()))
        return  params, param_props

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        lp += tfd.Normal(0, self.emission_weights_variance).log_prob(params['emissions']['weights']).sum()
        return lp

    def emission_distribution(self, params, state, **covariates):
        logits = params['emissions']['weights'][state] @ covariates['features']
        logits += params['emissions']['biases'][state]
        return tfd.Bernoulli(logits=logits)
