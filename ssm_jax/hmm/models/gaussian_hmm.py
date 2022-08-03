from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.abstractions import Parameter
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM
from ssm_jax.utils import PSDToRealBijector
from ssm_jax.distributions import NormalInverseWishart


@register_pytree_node_class
class GaussianHMM(StandardHMM):

    def __init__(self,
                 initial_probabilities,
                 transition_matrix,
                 emission_means,
                 emission_covariance_matrices,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self._emission_means = Parameter(emission_means)
        self._emission_covs = Parameter(emission_covariance_matrices, bijector=PSDToRealBijector)

        dim = emission_means.shape[-1]
        self._emission_prior_mean = Parameter(
            emission_prior_mean * jnp.ones(dim),
            is_frozen=True)
        self._emission_prior_conc = Parameter(
            emission_prior_concentration,
            is_frozen=True,
            bijector=tfb.Invert(tfb.Softplus()))
        self._emission_prior_scale = Parameter(
            emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
                else emission_prior_scale * jnp.eye(dim),
            is_frozen=True,
            bijector=PSDToRealBijector)
        self._emission_prior_df = Parameter(
            dim + emission_prior_extra_df,
            is_frozen=True,
            bijector=tfb.Invert(tfb.Softplus()))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self._emission_means

    @property
    def emission_covariance_matrices(self):
        return self._emission_covs

    def emission_distribution(self, state):
        return tfd.MultivariateNormalFullCovariance(self._emission_means.value[state],
                                                    self._emission_covs.value[state])

    def log_prior(self):
        lp = tfd.Dirichlet(self._initial_probs_concentration.value).log_prob(self.initial_probs.value)
        lp += tfd.Dirichlet(self._transition_matrix_concentration.value).log_prob(self.transition_matrix.value).sum()

        lp += NormalInverseWishart(
            self._emission_prior_mean.value,
            self._emission_prior_conc.value,
            self._emission_prior_df.value,
            self._emission_prior_scale.value
        ).log_prob((self.emission_covariance_matrices.value, self.emission_means.value)).sum()
        return lp

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class GaussianHMMSuffStats:
            # Wrapper for sufficient statistics of a GaussianHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_w: chex.Array
            sum_x: chex.Array
            sum_xxT: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(),
                                     self._compute_transition_matrices(),
                                     self._compute_conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix.value, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk,ti->ki", posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, emissions, emissions)

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = GaussianHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                         initial_probs=initial_probs,
                                         trans_probs=trans_probs,
                                         sum_w=sum_w,
                                         sum_x=sum_x,
                                         sum_xxT=sum_xxT)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, batch_emissions, batch_posteriors):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # The expected log joint is equal to the log prob of a normal inverse
        # Wishart distribution, up to additive factors. Find this NIW distribution
        # take its mode.
        mu0 = self._emission_prior_mean.value
        kappa0 = self._emission_prior_conc.value
        nu0 = self._emission_prior_df.value
        Psi0 = self._emission_prior_scale.value

        # Find the posterior parameters of the NIW distribution
        def _single_m_step(sum_w, sum_x, sum_xxT):
            kappa_post = kappa0 + sum_w
            mu_post = (kappa0 * mu0 + sum_x) / kappa_post
            nu_post = nu0 + sum_w
            Psi_post = Psi0 + kappa0 * jnp.outer(mu0, mu0) + sum_xxT - kappa_post * jnp.outer(mu_post, mu_post)
            return NormalInverseWishart(mu_post, kappa_post, nu_post, Psi_post).mode()

        covs, means = vmap(_single_m_step)(stats.sum_w, stats.sum_x, stats.sum_xxT)
        self.emission_covariance_matrices.value = covs
        self.emission_means.value = means
