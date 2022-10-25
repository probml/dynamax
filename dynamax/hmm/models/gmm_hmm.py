import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.scipy.special import logsumexp
from dynamax.parameters import ParameterProperties
from dynamax.distributions import NormalInverseWishart
from dynamax.distributions import niw_posterior_update
from dynamax.hmm.models.base import ExponentialFamilyHMM
from dynamax.utils import PSDToRealBijector


class GaussianMixtureHMM(ExponentialFamilyHMM):
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
                 emission_weights_concentration=1.1,
                 emission_prior_mean=0.,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_extra_df=1e-4,
                 emission_prior_scale=0.1):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)
        self.num_components = num_components
        self.emission_dim = emission_dim
        self.emission_weights_concentration = emission_weights_concentration * jnp.ones(num_components)
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_mean_concentration = emission_prior_mean_concentration
        self.emission_prior_df = emission_dim + emission_prior_extra_df
        self.emission_prior_scale = emission_prior_scale * jnp.eye(emission_dim)

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def _initialize_emissions(self, key):
        key1, key2 = jr.split(key, 2)
        weights = jr.dirichlet(key1, jnp.ones(self.num_components), shape=(self.num_states,))
        means = jr.normal(key2, (self.num_states, self.num_components, self.emission_dim))
        covs = jnp.eye(self.emission_dim) * jnp.ones((self.num_states, self.num_components, self.emission_dim, self.emission_dim))

        params = dict(weights=weights, means=means, covs=covs)
        param_props = dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
                           means=ParameterProperties(),
                           covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return  params, param_props

    # def kmeans_initialization(self, key, emissions):
    #     key0, key1, key2 = jr.split(key, 3)

    #     # https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py#L876
    #     main_kmeans = KMeans(n_clusters=self.num_states, random_state=42)
    #     covariance_matrix = None
    #     labels = main_kmeans.fit_predict(emissions)
    #     main_centroid = jnp.mean(main_kmeans.cluster_centers_, axis=0)
    #     emission_means = []
    #     keys = jr.split(key1, self.num_states)

    #     for label, key in enumerate(keys):
    #         cluster = emissions[jnp.where(labels == label)]
    #         needed = self.num_components
    #         centers = None

    #         if len(cluster):
    #             kmeans = KMeans(n_clusters=min(self.num_components, len(cluster)), random_state=label)
    #             kmeans.fit(np.array(cluster))
    #             centers = jnp.array(kmeans.cluster_centers_)
    #             needed = self.num_components - len(centers)

    #         if needed:
    #             if covariance_matrix is None:
    #                 covariance_matrix = jnp.cov(emissions.T) + 1e-6 * jnp.eye(self.emission_dim)
    #             random_centers = jr.multivariate_normal(key, main_centroid, cov=covariance_matrix, shape=(needed,))
    #             if centers is None:
    #                 emission_means.append(random_centers)
    #             else:
    #                 emission_means.append(jnp.vstack([centers, random_centers]))
    #         else:
    #             emission_means.append(centers)

    #     emission_means = jnp.array(emission_means)

    #     # Package into dictionaries
    #     initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
    #     transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
    #     emission_weights = jnp.ones(self.num_components) / self.num_components
    #     emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_components, 1, 1))

    #     params = dict(
    #         initial=dict(probs=initial_probs),
    #         transitions=dict(transition_matrix=transition_matrix),
    #         emissions=dict(weights=emission_weights, means=emission_means, covs=emission_covs))
    #     param_props = dict(
    #         initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
    #         transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
    #         emissions=dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
    #                        means=ParameterProperties(),
    #                        covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))))
    #     return  params, param_props

    # def kmeans_plusplus_initialization(self, key, emissions):
    #     key0, key1, key2 = jr.split(key, 3)

    #     centers, labels = kmeans_plusplus(np.array(emissions), n_clusters=self.num_states, random_state=42)

    #     main_centroid = jnp.mean(centers, axis=0)
    #     covariance_matrix = None

    #     emission_means = []
    #     keys = jr.split(key1, self.num_states)

    #     for label, key in enumerate(keys):
    #         cluster = emissions[jnp.where(labels == label)]
    #         needed = self.num_components
    #         centers = None

    #         if len(cluster):
    #             centers, _ = kmeans_plusplus(cluster, n_clusters=min(self.num_components, len(cluster)), random_state=label)
    #             centers = jnp.array(centers)
    #             needed = self.num_components - len(centers)

    #         if needed:
    #             if covariance_matrix is None:
    #                 covariance_matrix = jnp.cov(emissions.T) + 1e-6 * jnp.eye(self.emission_dim)
    #             random_centers = jr.multivariate_normal(key, main_centroid, cov=covariance_matrix, shape=(needed,))
    #             if centers is None:
    #                 emission_means.append(random_centers)
    #             else:
    #                 emission_means.append(jnp.vstack([centers, random_centers]))
    #         else:
    #             emission_means.append(centers)

    #     emission_means = jnp.array(emission_means)

    #     # Package into dictionaries
    #     initial_probs = jr.dirichlet(key1, jnp.ones(self.num_states))
    #     transition_matrix = jr.dirichlet(key2, jnp.ones(self.num_states), (self.num_states,))
    #     emission_weights = jnp.ones(self.num_components) / self.num_components
    #     emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_components, 1, 1))

    #     params = dict(
    #         initial=dict(probs=initial_probs),
    #         transitions=dict(transition_matrix=transition_matrix),
    #         emissions=dict(weights=emission_weights, means=emission_means, covs=emission_covs))
    #     param_props = dict(
    #         initial=dict(probs=ParameterProperties(constrainer=tfb.Softplus())),
    #         transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
    #         emissions=dict(weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
    #                        means=ParameterProperties(),
    #                        covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))))
    #     return  params, param_props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=params['emissions']['weights'][state]),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=params['emissions']['means'][state], covariance_matrix=params['emissions']['covs'][state]))

    def log_prior(self, params):
        lp = tfd.Dirichlet(self.initial_probs_concentration).log_prob(params['initial']['probs'])
        lp += tfd.Dirichlet(self.transition_matrix_concentration).log_prob(params['transitions']['transition_matrix']).sum()
        lp += tfd.Dirichlet(self.emission_weights_concentration).log_prob(
            params['emissions']['weights']).sum()
        lp += NormalInverseWishart(self.emission_prior_mean, self.emission_prior_mean_concentration,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params['emissions']['covs'], params['emissions']['means'])).sum()
        return lp

    def _zeros_like_suff_stats(self):
        return dict(N=jnp.zeros((self.num_states, self.num_components)),
                    Sx=jnp.zeros((self.num_states, self.num_components, self.emission_dim)),
                    SxxT=jnp.zeros((self.num_states, self.num_components, self.emission_dim, self.emission_dim)))

    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        def prob_fn(x):
            logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalFullCovariance(
                loc=mus, covariance_matrix=sigmas).log_prob(x) + jnp.log(weights))(
                    params['emissions']['means'], params['emissions']['covs'], params['emissions']['weights'])
            logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
            return jnp.exp(logprobs)

        prob_denses = vmap(prob_fn)(emissions)
        weights = jnp.einsum("tk,tkm->tkm", expected_states, prob_denses)
        Sx = jnp.einsum("tkm,tn->kmn", weights, emissions)
        SxxT = jnp.einsum("tkm,tn,tl->kmnl", weights, emissions, emissions)
        N = weights.sum(axis=0)

        return dict(N=N, Sx=Sx, SxxT=SxxT)

    def _m_step_emissions(self, params, param_props, emission_stats):
        assert param_props['emissions']['weights'].trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"
        assert param_props['emissions']['means'].trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"
        assert param_props['emissions']['covs'].trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"

        niw_prior = NormalInverseWishart(self.emission_prior_mean,
                                         self.emission_prior_mean_concentration,
                                         self.emission_prior_df,
                                         self.emission_prior_scale)

        def _single_m_step(Sx, SxxT, N):
            """Update the parameters for one discrete state"""
            # Update the component probabilities (i.e. weights)
            nu_post = self.emission_weights_concentration + N
            weights = tfd.Dirichlet(nu_post).mode()

            # Update the mean and covariance for each component
            covs, means = vmap(lambda stats: niw_posterior_update(niw_prior, stats).mode())((Sx, SxxT, N))
            return weights, means, covs

        weights, means, covs = vmap(_single_m_step)(
            emission_stats['Sx'], emission_stats['SxxT'], emission_stats['N'])
        params['emissions']['weights'] = weights
        params['emissions']['means'] = means
        params['emissions']['covs'] = covs
        return params
