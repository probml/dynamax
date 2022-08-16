import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.models.base import StandardHMM


class MultivariateNormalDiagHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_cov_diag_factors,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_cov_diag_factors_concentration=1.1,
                 emission_cov_diag_factors_rate=1.1):

        super().__init__(initial_probabilities, transition_matrix, initial_probs_concentration,
                         transition_matrix_concentration)
        self._emission_means = Parameter(emission_means)
        self._emission_cov_diag_factors = Parameter(emission_cov_diag_factors, bijector=tfb.Invert(tfb.Softplus()))

        # The hyperparameters of the prior
        self._emission_cov_diag_factors_concentration = Parameter(emission_cov_diag_factors_concentration *
                                                                  jnp.ones(self.num_states),
                                                                  is_frozen=True,
                                                                  bijector=tfb.Invert(tfb.Softplus()))
        self._emission_cov_diag_factors_rate = Parameter(emission_cov_diag_factors_rate * jnp.ones(self.num_states),
                                                         is_frozen=True,
                                                         bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_covs = jr.exponential(key4, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self._emission_means

    @property
    def emission_cov_diag_factors(self):
        return self._emission_cov_diag_factors

    def emission_distribution(self, state):
        return tfd.MultivariateNormalDiag(self._emission_means.value[state],
                                          self._emission_cov_diag_factors.value[state])

    @property
    def emission_distribution_parameters(self):
        return dict(
            emission_means=self._emission_means,
            emission_cov_diag_factors=self._emission_cov_diag_factors,
        )

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
        lp += tfd.Gamma(self._emission_cov_diag_factors_concentration.value,
                        self._emission_cov_diag_factors_rate.value).log_prob(
                            self.emission_cov_diag_factors.value).sum()
        return lp
