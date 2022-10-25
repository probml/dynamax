import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from dynamax.parameters import ParameterProperties
from dynamax.distributions import NormalInverseWishart
from dynamax.distributions import niw_posterior_update
from dynamax.hmm.models.base import ExponentialFamilyHMM
from dynamax.utils import PSDToRealBijector


class GaussianHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
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
        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
                else emission_prior_scale * jnp.eye(emission_dim)
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalFullCovariance(
            params['emissions']['means'][state], params['emissions']['covs'][state])

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params['emissions']['covs'], params['emissions']['means'])).sum()
        return lp

    def _initialize_emissions(self, key):
        emission_means = jr.normal(key, (self.num_states, self.emission_dim))
        emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        params = dict(means=emission_means, covs=emission_covs)
        param_props = dict(means=ParameterProperties(),
                           covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, param_props

    def _zeros_like_suff_stats(self):
        initial_stats = jnp.zeros(self.num_states)
        transition_stats = jnp.zeros((self.num_states, self.num_states))
        emission_stats = dict(
            sum_w=jnp.zeros((self.num_states,)),
            sum_x=jnp.zeros((self.num_states, self.emission_dim)),
            sum_xxT=jnp.zeros((self.num_states, self.emission_dim, self.emission_dim)),
        )
        return (initial_stats, transition_stats, emission_stats)

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        return dict(
            sum_w=jnp.einsum("tk->k", expected_states),
            sum_x=jnp.einsum("tk,ti->ki", expected_states, emissions),
            sum_xxT=jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        )

    def _m_step_emissions(self, params, param_props, emission_stats):
        if param_props['emissions']['covs'].trainable and \
            param_props['emissions']['means'].trainable:
            niw_prior = NormalInverseWishart(loc=self.emission_prior_mean,
                                            mean_concentration=self.emission_prior_conc,
                                            df=self.emission_prior_df,
                                            scale=self.emission_prior_scale)

            # Find the posterior parameters of the NIW distribution
            def _single_m_step(stats):
                niw_posterior = niw_posterior_update(niw_prior, (stats['sum_x'], stats['sum_xxT'], stats['sum_w']))
                return niw_posterior.mode()

            covs, means = vmap(_single_m_step)(emission_stats)
            params['emissions']['covs'] = covs
            params['emissions']['means'] = means

        elif param_props['emissions']['covs'].trainable and \
            not param_props['emissions']['means'].trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed means and trainable covariance")

        elif not param_props['emissions']['covs'].trainable and \
            param_props['emissions']['means'].trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed covariance and trainable means")

        return params
