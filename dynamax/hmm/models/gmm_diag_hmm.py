import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.scipy.special import logsumexp
from dynamax.parameters import ParameterProperties
from dynamax.distributions import NormalInverseGamma
from dynamax.distributions import nig_posterior_update
from dynamax.hmm.models.base import ExponentialFamilyHMM


class GaussianMixtureDiagHMM(ExponentialFamilyHMM):
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

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        emission_weights = jr.dirichlet(key1, jnp.ones(self.num_components), shape=(self.num_states,))
        emission_means = jr.normal(key2, (self.num_states, self.num_components, self.emission_dim))
        emission_scale_diags = jnp.ones((self.num_states, self.num_components, self.emission_dim))

        params = dict(weights=emission_weights,
                      means=emission_means,
                      scale_diags=emission_scale_diags)
        param_props = dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
                           means=ParameterProperties(),
                           scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
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
    def _zeros_like_suff_stats(self):
        return dict(N=jnp.zeros((self.num_states, self.num_components)),
                    Sx=jnp.zeros((self.num_states, self.num_components, self.emission_dim)),
                    Sxsq=jnp.zeros((self.num_states, self.num_components, self.emission_dim)))

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):

        # Evaluate the posterior probability of each discrete class
        def prob_fn(x):
            logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalDiag(
                loc=mus, scale_diag=sigmas).log_prob(x) + jnp.log(weights))(
                    params['emissions']['means'], params['emissions']['scale_diags'],
                    params['emissions']['weights'])
            logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
            return jnp.exp(logprobs)

        prob_denses = vmap(prob_fn)(emissions)
        weights = jnp.einsum("tk,tkm->tkm", expected_states, prob_denses)
        Sx = jnp.einsum("tkm,tn->kmn", weights, emissions)
        Sxsq = jnp.einsum("tkm,tn,tn->kmn", weights, emissions, emissions)
        N = weights.sum(axis=0)
        return dict(N=N, Sx=Sx, Sxsq=Sxsq)

    def _m_step_emissions(self, params, param_props, emission_stats):
        assert param_props['emissions']['weights'].trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"
        assert param_props['emissions']['means'].trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"
        assert param_props['emissions']['scale_diags'].trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"

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
        weights, means, scale_diags = vmap(_single_m_step)(
            emission_stats['Sx'], emission_stats['Sxsq'], emission_stats['N'])
        params['emissions']['weights'] = weights
        params['emissions']['means'] = means
        params['emissions']['scale_diags'] = scale_diags
        return params
