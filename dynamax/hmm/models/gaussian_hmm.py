import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from dynamax.parameters import ParameterProperties
from dynamax.distributions import InverseWishart
from dynamax.distributions import NormalInverseGamma
from dynamax.distributions import NormalInverseWishart
from dynamax.distributions import nig_posterior_update
from dynamax.distributions import niw_posterior_update
from dynamax.hmm.models.base import ExponentialFamilyHMM
from dynamax.hmm.models.base import StandardHMM
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

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_means=None,
                   emission_covariances=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_covariances (array, optional): manually specified emission covariances. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))

            _emission_means = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            this_key, key = jr.split(key)
            prior = NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                         self.emission_prior_df, self.emission_prior_scale)
            (_emission_covs, _emission_means) = prior.sample(seed=this_key, sample_shape=(self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(means=default(emission_means, _emission_means),
                                   covs=default(emission_covariances, _emission_covs))
        props['emissions'] = dict(means=ParameterProperties(),
                                  covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, props

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


class DiagonalGaussianHMM(ExponentialFamilyHMM):

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

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_means=None,
                   emission_scale_diags=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_scale_diags (array, optional): manually specified emission scales (sqrt of diagonal of covariance matrix). Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))

            _emission_means = jnp.array(km.cluster_centers_)
            _emission_scale_diags = jnp.ones((self.num_states, self.emission_dim))

        elif method.lower() == "prior":
            this_key, key = jr.split(key)
            prior = NormalInverseGamma(self.emission_prior_mean, self.emission_prior_conc,
                                       self.emission_prior_df, self.emission_prior_scale)
            (_emission_vars, _emission_means) = prior.sample(seed=this_key, sample_shape=(self.num_states,))
            _emission_scale_diags = jnp.sqrt(_emission_vars)

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(means=default(emission_means, _emission_means),
                                   scale_diags=default(emission_scale_diags, _emission_scale_diags))
        props['emissions'] = dict(means=ParameterProperties(),
                                  scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scale_diags'][state])

    def log_prior(self, params):
        lp = super().log_prior(params)
        prior =  NormalInverseGamma(self.emission_prior_mean, self.emission_prior_conc,
                                    self.emission_prior_df, self.emission_prior_scale)
        lp += prior.log_prob((params['emissions']['scale_diags']**2,
                              params['emissions']['means'])).sum()
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


class SphericalGaussianHMM(StandardHMM):

    def __init__(self,
                 num_states,
                 emission_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_mean_covariance=1.0,
                 emission_var_concentration=1.1,
                 emission_var_rate=1.1):

        super().__init__(num_states,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration)

        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_mean_cov = \
            emission_prior_mean_covariance if jnp.ndim(emission_prior_mean_covariance) == 2 \
                else emission_prior_mean_covariance * jnp.eye(emission_dim)
        self.emission_var_concentration = emission_var_concentration
        self.emission_var_rate = emission_var_rate

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_means=None,
                   emission_scales=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_scales (array, optional): manually specified emission scales (sqrt of diagonal of spherical covariance matrix). Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))

            _emission_means = jnp.array(km.cluster_centers_)
            _emission_scales = jnp.ones((self.num_states,))

        elif method.lower() == "prior":
            key1, key2, key = jr.split(key, 3)
            _emission_means = tfd.MultivariateNormalFullCovariance(
                self.emission_prior_mean, self.emission_prior_mean_cov)\
                    .sample(seed=key1, sample_shape=(self.num_states,))
            _emission_vars = tfd.Gamma(
                self.emission_var_concentration, self.emission_var_rate)\
                    .sample(seed=key2, sample_shape=(self.num_states,))
            _emission_scales = jnp.sqrt(_emission_vars)

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(means=default(emission_means, _emission_means),
                                   scales=default(emission_scales, _emission_scales))
        props['emissions'] = dict(means=ParameterProperties(),
                                  scales=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def emission_distribution(self, params, state, covariates=None):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params['emissions']['means'][state],
                                          params['emissions']['scales'][state] * jnp.ones((dim,)))

    def log_prior(self, params):
        lp = super().log_prior(params)
        lp += tfd.MultivariateNormalFullCovariance(
            self.emission_prior_mean, self.emission_prior_mean_cov)\
                .log_prob(params['emissions']['means']).sum()
        lp += tfd.Gamma(self.emission_var_concentration, self.emission_var_rate)\
            .log_prob(params['emissions']['scales']**2).sum()
        return lp


class SharedCovarianceGaussianHMM(ExponentialFamilyHMM):

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
            emission_covariance_matrix (_type_): _description_
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

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_means=None,
                   emission_covariance=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_covariance (array, optional): manually specified emission covariance. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Base class initializes the initial probs and transition matrix
        this_key, key = jr.split(key)
        params, props = super().initialize(key=this_key, method=method,
                                           initial_probs=initial_probs,
                                           transition_matrix=transition_matrix)

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))

            _emission_means = jnp.array(km.cluster_centers_)
            _emission_cov = jnp.eye(self.emission_dim)

        elif method.lower() == "prior":
            key1, key2, key = jr.split(key, 3)
            _emission_cov = InverseWishart(
                self.emission_prior_df, self.emission_prior_scale)\
                    .sample(seed=key1)
            _emission_means = tfd.MultivariateNormalFullCovariance(
                self.emission_prior_mean, self.emission_prior_conc * _emission_cov)\
                    .sample(seed=key2, sample_shape=(self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params['emissions'] = dict(means=default(emission_means, _emission_means),
                                   cov=default(emission_covariance, _emission_cov))
        props['emissions'] = dict(means=ParameterProperties(),
                                  cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, props

    def emission_distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalFullCovariance(
            params['emissions']['means'][state], params['emissions']['cov'])

    def log_prior(self, params):
        lp = super().log_prior(params)

        mus = params['emissions']['means']
        Sigma = params['emissions']['cov']
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        lp += InverseWishart(nu0, Psi0).log_prob(Sigma)
        lp += tfd.MultivariateNormalFullCovariance(mu0, Sigma / kappa0).log_prob(mus).sum()
        return lp

    def _zeros_like_suff_stats(self):
        dim = self.num_obs
        num_states = self.num_states
        return dict(
            sum_w=jnp.zeros((num_states,)),
            sum_x=jnp.zeros((num_states, dim)),
            sum_xxT=jnp.zeros((num_states, dim, dim)),
        )

    # Expectation-maximization (EM) code
    def _compute_expected_suff_stats(self, params, emissions, expected_states, covariates=None):
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("ti,tj->ij", emissions, emissions)
        sum_T = len(emissions)
        stats = dict(sum_w=sum_w, sum_x=sum_x, sum_xxT=sum_xxT, sum_T=sum_T)
        return stats

    def _m_step_emissions(self, params, param_props, emission_stats):
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        sum_T = emission_stats['sum_T'] + nu0 + self.num_states + self.emission_dim + 1
        sum_w = emission_stats['sum_w'] + kappa0
        sum_x = emission_stats['sum_x'] + kappa0 * mu0
        sum_xxT = emission_stats['sum_xxT'] + Psi0 + kappa0 * jnp.outer(mu0, mu0)
        params['emissions']['means'] = jnp.einsum('ki,k->ki', sum_x, 1/sum_w)
        params['emissions']['cov'] = (sum_xxT - jnp.einsum('ki,kj,k->ij', sum_x, sum_x, 1/sum_w)) / sum_T
        return params
