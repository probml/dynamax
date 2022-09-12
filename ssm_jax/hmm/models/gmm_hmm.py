from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import tree_map
from jax import vmap
from jax.scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from ssm_jax.parameters import ParameterProperties
from ssm_jax.distributions import NormalInverseWishart
from ssm_jax.distributions import niw_posterior_update
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import StandardHMM
from ssm_jax.utils import PSDToRealBijector


@chex.dataclass
class GMMHMMSuffStats:
    # Wrapper for sufficient statistics of a GMMHMM
    marginal_loglik: chex.Scalar
    initial_probs: chex.Array
    trans_probs: chex.Array
    N: chex.Array
    Sx: chex.Array
    SxxT: chex.Array


class GaussianMixtureHMM(StandardHMM):
    """
    Hidden Markov Model with Gaussian mixture emissions.
    Attributes
    ----------
    weights : array, shape (num_states, num_emission_components)
        Mixture weights for each state.
    emission_means : array, shape (num_states, num_emission_components, emission_dim)
        Mean parameters for each mixture component in each state.
    emission_covariance_matrices : array
        Covariance parameters for each mixture components in each state.
    """

    def __init__(self,
                 num_states,
                 num_components,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_mixture_weights_concentration=1.1,
                 emission_prior_mean=0.,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_extra_df=1e-4,
                 emission_prior_scale=0.1):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.num_components = num_components
        self.emission_dim = emission_dim
        self.emission_mixture_weights_concentration = emission_mixture_weights_concentration
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_mean_concentration = emission_prior_mean_concentration
        self.emission_prior_df = emission_dim + emission_prior_extra_df
        self.emission_prior_scale = emission_prior_scale * jnp.eye(emission_dim)

    def random_initialization(self, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
        emission_weights = jr.dirichlet(key3, jnp.ones(self.num_components), shape=(self.num_states,))
        emission_means = jr.normal(key4, (self.num_states, self.num_components, self.emission_dim))
        emission_covs = jnp.eye(self.emission_dim) * jnp.ones((self.num_states, self.num_components, self.emission_dim, self.emission_dim))

        params = dict(
            initial=dict(probs=initial_probs),
            transitions=dict(transition_matrix=transition_matrix),
            emissions=dict(weights=emission_weights, means=emission_means, covs=emission_covs))
        param_props = dict(
            initial=dict(probs=ParameterProperties(constrainer=tfb.Sotfplus())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
                           means=ParameterProperties(),
                           covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))))
        return  params, param_props

    def kmeans_initialization(self, key, emissions):
        key0, key1 = jr.split(key)

        params, param_props = self.random_initialization(key0)
        # https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py#L876
        main_kmeans = KMeans(n_clusters=self.num_states, random_state=42)
        covariance_matrix = None
        labels = main_kmeans.fit_predict(emissions)
        main_centroid = jnp.mean(main_kmeans.cluster_centers_, axis=0)
        emission_means = []
        keys = jr.split(key1, self.num_states)

        for label, key in enumerate(keys):
            cluster = emissions[jnp.where(labels == label)]
            needed = num_components
            centers = None

            if len(cluster):
                kmeans = KMeans(n_clusters=min(num_components, len(cluster)), random_state=label)
                kmeans.fit(np.array(cluster))
                centers = jnp.array(kmeans.cluster_centers_)
                needed = num_components - len(centers)

            if needed:
                if covariance_matrix is None:
                    covariance_matrix = jnp.cov(emissions.T) + 1e-6 * jnp.eye(emission_dim)
                random_centers = jr.multivariate_normal(key, main_centroid, cov=covariance_matrix, shape=(needed,))
                if centers is None:
                    emission_means.append(random_centers)
                else:
                    emission_means.append(jnp.vstack([centers, random_centers]))
            else:
                emission_means.append(centers)

        emission_means = jnp.array(emission_means)
        hmm._emission_means.value = emission_means
        return hmm

    @classmethod
    def kmeans_plusplus_initialization(cls, key, num_states, num_components, emission_dim, emissions):
        key0, key1 = jr.split(key)
        hmm = GaussianMixtureHMM.random_initialization(key0, num_states, num_components, emission_dim)

        centers, labels = kmeans_plusplus(np.array(emissions), n_clusters=num_states, random_state=42)

        main_centroid = jnp.mean(centers, axis=0)
        covariance_matrix = None

        emission_means = []
        keys = jr.split(key1, num_states)

        for label, key in enumerate(keys):
            cluster = emissions[jnp.where(labels == label)]
            needed = num_components
            centers = None

            if len(cluster):
                centers, _ = kmeans_plusplus(cluster, n_clusters=min(num_components, len(cluster)), random_state=label)
                centers = jnp.array(centers)
                needed = num_components - len(centers)

            if needed:
                if covariance_matrix is None:
                    covariance_matrix = jnp.cov(emissions.T) + 1e-6 * jnp.eye(emission_dim)
                random_centers = jr.multivariate_normal(key, main_centroid, cov=covariance_matrix, shape=(needed,))
                if centers is None:
                    emission_means.append(random_centers)
                else:
                    emission_means.append(jnp.vstack([centers, random_centers]))
            else:
                emission_means.append(centers)

        emission_means = jnp.array(emission_means)
        hmm._emission_means.value = emission_means
        return hmm

    def emission_distribution(self, params, state):
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=params['emissions']['weights'][state]),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=params['emissions']['means'][state], covariance_matrix=params['emissions']['covs'][state]))

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(params['transitions']['transition_matrix']).sum()
        lp += tfd.Dirichlet(self.emission_prior_weights_concentration).log_prob(
            params['emissions']['weights']).sum()
        lp += NormalInverseWishart(self.emission_prior_mean, self.emission_prior_mean_concentration,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params['emissions']['covs'], params['emissions']['means'])).sum()
        return lp

    # Expectation-maximization (EM) code
    def e_step(self, params, batch_emissions, **batch_covariates):

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self._compute_initial_probs(params), self._compute_transition_matrices(params),
                                     self._compute_conditional_logliks(params, emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(params['transitions']['transition_matrix'], posterior)

            def prob_fn(x):
                logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalFullCovariance(
                    loc=mus, covariance_matrix=sigmas).log_prob(x) + jnp.log(weights))(
                        params['emissions']['means'], params['emissions']['covs'], params['emissions']['weights'])
                logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
                return jnp.exp(logprobs)

            prob_denses = vmap(prob_fn)(emissions)
            N = jnp.einsum("tk,tkm->tkm", posterior.smoothed_probs, prob_denses)
            Sx = jnp.einsum("tkm,tn->kmn", N, emissions)
            SxxT = jnp.einsum("tkm,tn,tl->kmnl", N, emissions, emissions)
            N = N.sum(axis=0)

            stats = GMMHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                    initial_probs=initial_probs,
                                    trans_probs=trans_probs,
                                    N=N,
                                    Sx=Sx,
                                    SxxT=SxxT)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def _m_step_emissions(self, params, param_props, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        def _single_m_step(Sx, SxxT, N):
            nu_post = self.emission_mixture_weights_concentration + N
            weights = tfd.Dirichlet(nu_post).mode()

            niw_prior = NormalInverseWishart(self.emission_prior_mean,
                                             self.emission_prior_mean_concentration,
                                             self.emission_prior_df,
                                             self.emiss)
            covs, means = niw_posterior_update(niw_prior, (Sx, SxxT, N)).mode()
            return weights, means, covs

        weights, means, covs = vmap(_single_m_step)(stats.Sx, stats.SxxT, stats.N)
        params['emissions']['weights'] = weights
        params['emissions']['means'] = means
        params['emissions']['covs'] = covs
