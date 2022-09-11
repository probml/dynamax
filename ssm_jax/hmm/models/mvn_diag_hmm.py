from functools import partial

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from ssm_jax.parameters import ParameterProperties
from ssm_jax.distributions import NormalInverseGamma
from ssm_jax.distributions import nig_posterior_update
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM
from ssm_jax.hmm.models.gaussian_hmm import GaussianHMMSuffStats


class MultivariateNormalDiagHMM(StandardHMM):

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

    def random_initialization(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_means = jr.normal(key3, (self.num_states, self.emission_dim))
        emission_scale_diags = jr.exponential(key4, (self.num_states, self.emission_dim))
        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(means=emission_means, scale_diags=emission_scale_diags))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Sotfplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), scale_diags=ParameterProperties(constrainer=tfb.Softplus())))
        return  params, param_props

    def emission_distribution(self, params, state):
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scale_diags'][state])

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(
            params['transitions']['transition_matrix']).sum()
        lp += NormalInverseGamma(
            self.emission_prior_mean,
            self.emission_prior_conc,
            self.emission_prior_df,
            self.emission_prior_scale,
        ).log_prob((params['emissions']['scale_diags'], params['emissions']['means'])).sum()
        return lp

    # Expectation-maximization (EM) code
    def e_step(self, params, batch_emissions):

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(params),
                                     self._compute_transition_matrices(params),
                                     self._compute_conditional_logliks(params, emissions))

            # Compute the initial state and transition probabilities
            trans_probs = compute_transition_probs(params['transitions']['transition_matrix'], posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk,ti->ki", posterior.smoothed_probs, emissions)
            sum_x2 = jnp.einsum("tk,ti,ti->ki", posterior.smoothed_probs, emissions, emissions)

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = GaussianHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                         initial_probs=posterior.initial_probs,
                                         trans_probs=trans_probs,
                                         sum_w=sum_w,
                                         sum_x=sum_x,
                                         sum_xxT=sum_x2)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        nig_prior = NormalInverseGamma(loc=self.emission_prior_mean,
                                       mean_concentration=self.emission_prior_conc,
                                       concentration=self.emission_prior_df,
                                       scale=self.emission_prior_scale)

        # The expected log joint is equal to the log prob of a normal inverse
        # gamma distribution, up to additive factors. Find this NIG distribution
        # take its mode.
        def _single_m_step(*stats):
            # Find the posterior parameters of the NIG distribution
            posterior = nig_posterior_update(nig_prior, stats)
            return posterior.mode()

        vars, means = vmap(_single_m_step)(stats.sum_x, stats.sum_xxT, stats.sum_w)
        params['emissions']['scale_diags'] = jnp.sqrt(vars)
        params['emissions']['means'] = means
        return params
