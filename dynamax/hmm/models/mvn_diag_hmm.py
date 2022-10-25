import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from dynamax.parameters import ParameterProperties
from dynamax.distributions import NormalInverseGamma
from dynamax.distributions import nig_posterior_update
from dynamax.hmm.models.base import ExponentialFamilyHMM


class MultivariateNormalDiagHMM(ExponentialFamilyHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale * jnp.ones(emission_dim) \
            if isinstance(emission_prior_scale, float) else emission_prior_scale
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        emission_means = jr.normal(key1, (self.num_states, self.emission_dim))
        emission_scale_diags = jr.exponential(key2, (self.num_states, self.emission_dim))
        params = dict(means=emission_means, scale_diags=emission_scale_diags)
        param_props = dict(means=ParameterProperties(), scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scale_diags'][state])

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += NormalInverseGamma(
            self.emission_prior_mean,
            self.emission_prior_conc,
            self.emission_prior_df,
            self.emission_prior_scale,
        ).log_prob((params['emissions']['scale_diags'], params['emissions']['means'])).sum()
        return lp

    def _zeros_like_suff_stats(self):
        return dict(sum_w=jnp.zeros(self.num_states),
                    sum_x=jnp.zeros((self.num_states, self.emission_dim)),
                    sum_xsq=jnp.zeros((self.num_states, self.emission_dim)))

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xsq = jnp.einsum("tk,ti->ki", expected_states, emissions**2)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_xsq=sum_xsq)

    def _m_step_emissions(self, params, param_props, emission_stats):
        nig_prior = NormalInverseGamma(loc=self.emission_prior_mean,
                                       mean_concentration=self.emission_prior_conc,
                                       concentration=self.emission_prior_df,
                                       scale=self.emission_prior_scale)

        def _single_m_step(stats):
            # Find the posterior parameters of the NIG distribution
            posterior = nig_posterior_update(nig_prior, (stats['sum_x'], stats['sum_xsq'], stats['sum_w']))
            return posterior.mode()

        vars, means = vmap(_single_m_step)(emission_stats)
        params['emissions']['scale_diags'] = jnp.sqrt(vars)
        params['emissions']['means'] = means
        return params
