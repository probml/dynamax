import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties
from dynamax.hmm.models.base import StandardHMM


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

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        emission_means = jr.normal(key1, (self.num_states, self.emission_dim))
        emission_scales = jr.exponential(key2, (self.num_states, 1))
        params = dict(means=emission_means, scales=emission_scales)
        param_props = dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus()))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scales'][state] * jnp.ones((dim,)))

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.Gamma(self.emission_var_concentration,
                        self.emission_var_rate).log_prob(params['emissions']['scales']**2).sum()
        return lp
