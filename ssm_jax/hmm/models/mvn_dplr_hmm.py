import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.tree_util import register_pytree_node_class
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.models.base import StandardHMM


@register_pytree_node_class
class MultivariateNormalDiagPlusLowRankHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_cov_diag_factors,
                 emission_cov_low_rank_factors,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self._emission_means = Parameter(emission_means)
        self._emission_cov_diag_factors = Parameter(emission_cov_diag_factors, bijector=tfb.Invert(tfb.Softplus()))
        self._emission_cov_low_rank_factors = Parameter(emission_cov_low_rank_factors)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim, emission_cov_rank):
        key1, key2, key3, key4, key5 = jr.split(key, 5)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_cov_diag_factors = jr.exponential(key4, (num_states, emission_dim))
        emission_cov_low_rank_factors = jr.normal(key5, (num_states, emission_dim, emission_cov_rank))
        return cls(initial_probs, transition_matrix, emission_means,
                   emission_cov_diag_factors, emission_cov_low_rank_factors)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self._emission_means

    @property
    def emission_cov_diag_factors(self):
        return self._emission_cov_diag_factors

    @property
    def emission_cov_low_rank_factors(self):
        return self._emission_cov_low_rank_factors

    def emission_distribution(self, state):
        return tfd.MultivariateNormalDiagPlusLowRankCovariance(
            self.emission_means.value[state],
            self.emission_cov_diag_factors.value[state],
            self.emission_cov_low_rank_factors.value[state]
        )

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
        lp += tfd.Gamma(1.1, 1.1).log_prob(self.emission_cov_diag_factors.value).sum()
        return lp
