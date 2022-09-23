from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.scipy.special import logsumexp
from ssm_jax.parameters import ParameterProperties
from ssm_jax.distributions import NormalInverseGamma
from ssm_jax.distributions import nig_posterior_update
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM


@chex.dataclass
class GMMDiagHMMSuffStats:
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    N: chex.Array
    Sx: chex.Array
    Sxsq: chex.Array


class GaussianMixtureDiagHMM(StandardHMM):
    """
    Hidden Markov Model with Gaussian mixture emissions where covariance matrices are diagonal.
    Attributes
    ----------
    weights : array, shape (num_states, num_emission_components)
        Mixture weights for each state.
    emission_means : array, shape (num_states, num_emission_components, emission_dim)
        Mean parameters for each mixture component in each state.
    emission_cov_diag_factors : array
        Diagonal entities of covariance parameters for each mixture components in each state.
    Remark
    ------
    Inverse gamma distribution has two parameters which are shape and scale.
    So, emission_prior_shape variable has nothing to do with the shape of any array.
    """

    def __init__(self,
                 num_states,
                 num_components,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_weights_concentration=1.1,
                 emission_prior_mean=0.,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_shape=1.,
                 emission_prior_scale=1.):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.num_components = num_components
        self.emission_dim = emission_dim

        self.emission_prior_weights_concentration = \
            emission_prior_weights_concentration * jnp.ones(num_components)

        self.emission_prior_mean = emission_prior_mean
        self.emission_prior_mean_concentration = emission_prior_mean_concentration
        self.emission_prior_shape = emission_prior_shape
        self.emission_prior_scale = emission_prior_scale

    def random_initialization(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_weights = jr.dirichlet(key3, jnp.ones(self.num_components), shape=(self.num_states,))
        emission_means = jr.normal(key4, (self.num_states, self.num_components, self.emission_dim))
        emission_scale_diags = jnp.ones((self.num_states, self.num_components, self.emission_dim))

        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(weights=emission_weights, means=emission_means, scale_diags=emission_scale_diags))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
                           means=ParameterProperties(),
                           scale_diags=ParameterProperties(constrainer=tfb.Softplus())))
        return  params, param_props

    def emission_distribution(self, params, state):
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=params['emissions']['weights'][state]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=params['emissions']['means'][state],
                scale_diag=params['emissions']['scale_diags'][state]))

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(params['transitions']['transition_matrix']).sum()
        lp += tfd.Dirichlet(self.emission_prior_weights_concentration).log_prob(
            params['emissions']['weights']).sum()
        lp += NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_concentration,
                                   self.emission_prior_shape, self.emission_prior_scale).log_prob(
            (params['emissions']['scale_diags']**2, params['emissions']['means'])).sum()
        return lp

    # Expectation-maximization (EM) code
    def e_step(self, params, batch_emissions, **batch_covariates):

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(params),
                                     self._compute_transition_matrices(params),
                                     self._compute_conditional_logliks(params, emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(params['transitions']['transition_matrix'], posterior)

            # Evaluate the posterior probability of each discrete class
            def prob_fn(x):
                logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalDiag(
                    loc=mus, scale_diag=sigmas).log_prob(x) + jnp.log(weights))(
                        params['emissions']['means'], params['emissions']['scale_diags'],
                        params['emissions']['weights'])
                logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
                return jnp.exp(logprobs)

            prob_denses = vmap(prob_fn)(emissions)
            N = jnp.einsum("tk,tkm->tkm", posterior.smoothed_probs, prob_denses)
            Sx = jnp.einsum("tkm,tn->kmn", N, emissions)
            Sxsq = jnp.einsum("tkm,tn,tn->kmn", N, emissions, emissions)
            N = N.sum(axis=0)

            stats = GMMDiagHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                        initial_probs=initial_probs,
                                        trans_probs=trans_probs,
                                        N=N,
                                        Sx=Sx,
                                        Sxsq=Sxsq)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, param_props, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        nig_prior = NormalInverseGamma(
            self.emission_prior_mean, self.emission_prior_mean_concentration,
            self.emission_prior_shape, self.emission_prior_scale)

        def _single_m_step(Sx, Sxsq, N):
            """Update the parameters for one discrete state"""
            # Update the component probabilities (i.e. weights)
            nu_post = self.emission_prior_weights_concentration + N
            mixture_weights = tfd.Dirichlet(nu_post).mode()

            # Update the mean and variances for each component
            var_diags, means = vmap(lambda stats: nig_posterior_update(nig_prior, stats).mode())((Sx, Sxsq, N))
            scale_diags = jnp.sqrt(var_diags)
            return mixture_weights, means, scale_diags

        # Compute mixture weights, diagonal factors of covariance matrices and means
        # for each state in parallel. Note that the first dimension of all sufficient
        # statistics is equal to number of states of HMM.
        weights, means, scale_diags = vmap(_single_m_step)(stats.Sx, stats.Sxsq, stats.N)
        params['emissions']['weights'] = weights
        params['emissions']['means'] = means
        params['emissions']['scale_diags'] = scale_diags
        return params
