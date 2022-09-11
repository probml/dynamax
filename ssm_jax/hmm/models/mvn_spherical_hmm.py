import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from ssm_jax.parameters import ParameterProperties
from ssm_jax.hmm.models.base import StandardHMM


class MultivariateNormalSphericalHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_var_concentration=1.1,
                 emission_var_rate=1.1):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_dim
        self.emission_var_concentration = emission_var_concentration
        self.emission_var_rate = emission_var_rate

    def random_initialization(self, key):
        key1, key2, key3, key4, key5 = jr.split(key, 5)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_means = jr.normal(key3, (self.num_states, self.emission_dim))
        emission_scales = jr.exponential(key4, (self.num_states, 1))
        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(means=emission_means, scales=emission_scales))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Sotfplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus())))
        return  params, param_props

    def emission_distribution(self, params, state):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scales'][state] * jnp.ones((dim,)))

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        lp += tfd.Gamma(self.emission_var_concentration.value,
                         self.emission_var_rate.value).log_prob(params['emissions']['scales']**2).sum()
        return lp
