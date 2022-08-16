from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM
from ssm_jax.utils import PSDToRealBijector


@register_pytree_node_class
class LinearRegressionHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_matrices,
                 emission_biases,
                 emission_covariance_matrices,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_matrices (_type_): _description_
            emission_biases (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities,
                         transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self._emission_matrices = Parameter(emission_matrices)
        self._emission_biases = Parameter(emission_biases)
        self._emission_covs = Parameter(emission_covariance_matrices, bijector=PSDToRealBijector)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim, feature_dim):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_matrices = jr.normal(key3, (num_states, emission_dim, feature_dim))
        emission_biases = jr.normal(key4, (num_states, emission_dim))
        emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, emission_matrices, emission_biases, emission_covs)

    # Properties to get various parameters of the model
    @property
    def emission_matrices(self):
        return self._emission_matrices

    @property
    def emission_biases(self):
        return self._emission_biases

    @property
    def emission_covariance_matrices(self):
        return self._emission_covs

    def emission_distribution(self, state, **covariates):
        prediction = self._emission_matrices.value[state] @ covariates['features'] + self._emission_biases.value[state]
        return tfd.MultivariateNormalFullCovariance(prediction, self._emission_covs.value[state])

    @property
    def emission_distribution_parameters(self):
        return dict(emission_matrices=self._emission_matrices,
                    emission_biases=self._emission_biases,
                    emission_covariance_matrices=self._emission_covs)

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()
        # TODO: Add MatrixNormalInverseWishart prior
        return lp

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions, **batch_covariates):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class LinearRegressionHMMSuffStats:
            # Wrapper for sufficient statistics of a GaussianHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_w: chex.Array
            sum_x: chex.Array
            sum_y: chex.Array
            sum_xxT: chex.Array
            sum_xyT: chex.Array
            sum_yyT: chex.Array

        def _single_e_step(emissions, **covariates):
            features = covariates['features']
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(), self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions, features=features))

            # Compute the initial state and transition probabilities
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk,ti->ki", posterior.smoothed_probs, features)
            sum_y = jnp.einsum("tk,ti->ki", posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, features, features)
            sum_xyT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, features, emissions)
            sum_yyT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, emissions, emissions)

            return LinearRegressionHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                                initial_probs=posterior.initial_probs,
                                                trans_probs=trans_probs,
                                                sum_w=sum_w,
                                                sum_x=sum_x,
                                                sum_y=sum_y,
                                                sum_xxT=sum_xxT,
                                                sum_xyT=sum_xyT,
                                                sum_yyT=sum_yyT)

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions, **batch_covariates)

    def _m_step_emissions(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # TODO: Add MatrixNormalInverseWishart prior

        # Find the posterior parameters of the NIW distribution
        def _single_m_step(sum_w, sum_x, sum_y, sum_xxT, sum_xyT, sum_yyT):
            # Make block matrices for stacking features (x) and bias (1)
            sum_x1x1T = jnp.block([[sum_xxT, jnp.expand_dims(sum_x, 1)],
                                   [jnp.expand_dims(sum_x, 0),
                                    jnp.expand_dims(sum_w, (0, 1))]])
            sum_x1yT = jnp.vstack([sum_xyT, sum_y])

            # Solve for the optimal A, b, and Sigma
            Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
            Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
            Sigma = 0.5 * (Sigma + Sigma.T)  # for numerical stability
            return Ab[:, :-1], Ab[:, -1], Sigma

        As, bs, Sigmas = vmap(_single_m_step)(stats.sum_w, stats.sum_x, stats.sum_y, stats.sum_xxT, stats.sum_xyT,
                                              stats.sum_yyT)
        self.emission_matrices.value = As
        self.emission_biases.value = bs
        self.emission_covariance_matrices.value = Sigmas
