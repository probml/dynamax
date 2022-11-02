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
from dynamax.hmm.models.abstractions import HMM, HMMEmissions
from dynamax.hmm.models.initial import StandardHMMInitialState
from dynamax.hmm.models.transitions import StandardHMMTransitions
from dynamax.utils import PSDToRealBijector, pytree_sum


class GaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
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
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
                else emission_prior_scale * jnp.eye(emission_dim)
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalFullCovariance(
            params['means'][state], params['covs'][state])

    def log_prior(self, params):
        return NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params['covs'], params['means'])).sum()

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
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
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_covariances (array, optional): manually specified emission covariances. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
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
        params = dict(means=default(emission_means, _emission_means),
                      covs=default(emission_covariances, _emission_covs))
        props = dict(means=ParameterProperties(),
                     covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, covariates=None):
        expected_states = posterior.smoothed_probs
        return dict(
            sum_w=jnp.einsum("tk->k", expected_states),
            sum_x=jnp.einsum("tk,ti->ki", expected_states, emissions),
            sum_xxT=jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        )

    def m_step(self, params, props, batch_stats):
        if props['covs'].trainable and props['means'].trainable:
            niw_prior = NormalInverseWishart(loc=self.emission_prior_mean,
                                            mean_concentration=self.emission_prior_conc,
                                            df=self.emission_prior_df,
                                            scale=self.emission_prior_scale)

            # Find the posterior parameters of the NIW distribution
            def _single_m_step(stats):
                niw_posterior = niw_posterior_update(niw_prior, (stats['sum_x'], stats['sum_xxT'], stats['sum_w']))
                return niw_posterior.mode()

            emission_stats = pytree_sum(batch_stats, axis=0)
            params['covs'], params['means'] = vmap(_single_m_step)(emission_stats)

        elif props['covs'].trainable and not props['means'].trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed means and trainable covariance")

        elif not props['covs'].trainable and props['means'].trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed covariance and trainable means")

        return params


class DiagonalGaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_concentration=1.1,
                 emission_prior_scale=1.1):

        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_mean_conc = emission_prior_mean_concentration
        self.emission_prior_conc = emission_prior_concentration * jnp.ones(emission_dim) \
            if isinstance(emission_prior_concentration, float) else emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
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
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_scale_diags (array, optional): manually specified emission scales (sqrt of diagonal of covariance matrix). Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))
            _emission_means = jnp.array(km.cluster_centers_)
            _emission_scale_diags = jnp.ones((self.num_states, self.emission_dim))

        elif method.lower() == "prior":
            this_key, key = jr.split(key)
            prior = NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_conc,
                                       self.emission_prior_scale, self.emission_prior_conc)
            (_emission_vars, _emission_means) = prior.sample(seed=this_key, sample_shape=(self.num_states,))
            _emission_scale_diags = jnp.sqrt(_emission_vars)

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = dict(means=default(emission_means, _emission_means),
                      scale_diags=default(emission_scale_diags, _emission_scale_diags))
        props = dict(means=ParameterProperties(),
                     scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalDiag(params['means'][state],
                                          params['scale_diags'][state])

    def log_prior(self, params):
        prior =  NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_conc,
                                    self.emission_prior_conc, self.emission_prior_scale)
        return prior.log_prob((params['scale_diags']**2,
                               params['means'])).sum()

    def collect_suff_stats(self, params, posterior, emissions, covariates=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xsq = jnp.einsum("tk,ti->ki", expected_states, emissions**2)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_xsq=sum_xsq)

    def m_step(self, params, props, batch_stats):
        nig_prior = NormalInverseGamma(loc=self.emission_prior_mean,
                                       mean_concentration=self.emission_prior_mean_conc,
                                       concentration=self.emission_prior_conc,
                                       scale=self.emission_prior_scale)

        def _single_m_step(stats):
            # Find the posterior parameters of the NIG distribution
            posterior = nig_posterior_update(nig_prior, (stats['sum_x'], stats['sum_xsq'], stats['sum_w']))
            return posterior.mode()

        emission_stats = pytree_sum(batch_stats, axis=0)
        vars, means = vmap(_single_m_step)(emission_stats)
        params['scale_diags'] = jnp.sqrt(vars)
        params['means'] = means
        return params


class SphericalGaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_mean_covariance=1.0,
                 emission_var_concentration=1.1,
                 emission_var_rate=1.1):
        self.num_states = num_states
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
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_scales (array, optional): manually specified emission scales (sqrt of diagonal of spherical covariance matrix). Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
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
        params = dict(means=default(emission_means, _emission_means),
                      scales=default(emission_scales, _emission_scales))
        props = dict(means=ParameterProperties(),
                     scales=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, params, state, covariates=None):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params['means'][state],
                                          params['scales'][state] * jnp.ones((dim,)))

    def log_prior(self, params):
        lp = tfd.MultivariateNormalFullCovariance(
            self.emission_prior_mean, self.emission_prior_mean_cov)\
                .log_prob(params['means']).sum()
        lp += tfd.Gamma(self.emission_var_concentration, self.emission_var_rate)\
            .log_prob(params['scales']**2).sum()
        return lp


class SharedCovarianceGaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):
        """_summary_

        Args:
            emission_means (_type_): _description_
            emission_covariance_matrix (_type_): _description_
        """
        self.num_states = num_states
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
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_covariance (array, optional): manually specified emission covariance. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
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
        params = dict(means=default(emission_means, _emission_means),
                      cov=default(emission_covariance, _emission_cov))
        props = dict(means=ParameterProperties(),
                     cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, props

    def distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalFullCovariance(
            params['means'][state], params['cov'])

    def log_prior(self, params):
        mus = params['means']
        Sigma = params['cov']
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        lp = InverseWishart(nu0, Psi0).log_prob(Sigma)
        lp += tfd.MultivariateNormalFullCovariance(mu0, Sigma / kappa0).log_prob(mus).sum()
        return lp

    def collect_suff_stats(self, params, posterior, emissions, covariates=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("ti,tj->ij", emissions, emissions)
        sum_T = len(emissions)
        stats = dict(sum_w=sum_w, sum_x=sum_x, sum_xxT=sum_xxT, sum_T=sum_T)
        return stats

    def m_step(self, params, props, batch_stats):
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        emission_stats = pytree_sum(batch_stats, axis=0)
        sum_T = emission_stats['sum_T'] + nu0 + self.num_states + self.emission_dim + 1
        sum_w = emission_stats['sum_w'] + kappa0
        sum_x = emission_stats['sum_x'] + kappa0 * mu0
        sum_xxT = emission_stats['sum_xxT'] + Psi0 + kappa0 * jnp.outer(mu0, mu0)
        params['means'] = jnp.einsum('ki,k->ki', sum_x, 1/sum_w)
        params['cov'] = (sum_xxT - jnp.einsum('ki,kj,k->ij', sum_x, sum_x, 1/sum_w)) / sum_T
        return params


class LowRankGaussianHMMEmissions(HMMEmissions):

    def __init__(self, num_states, emission_dim, emission_rank,
                 emission_diag_factor_concentration=1.1,
                 emission_diag_factor_rate=1.1):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_rank = emission_rank
        self.emission_diag_factor_conc = emission_diag_factor_concentration
        self.emission_diag_factor_rate = emission_diag_factor_rate

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   emission_means=None,
                   emission_cov_diag_factors=None,
                   emission_cov_low_rank_factors=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_means (array, optional): manually specified emission means. Defaults to None.
            emission_cov_diag_factors (array, optional): manually specified diagonals of the emission covariances. Defaults to None.
            emission_cov_low_rank_factors (array, optional): manually specified low rank factors of the emission covariances. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: a nested dictionary of arrays containing the model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))
            _emission_means = jnp.array(km.cluster_centers_)
            _emission_cov_diag_factors = jnp.ones((self.num_states, self.emission_dim))
            _emission_cov_low_rank_factors = jnp.zeros((self.num_states, self.emission_dim, self.emission_rank))

        elif method.lower() == "prior":
            # We don't have a real prior
            key1, key2, key3 = jr.split(key, 3)
            _emission_means = jr.normal(key1, (self.num_states, self.emission_dim))
            _emission_cov_diag_factors = \
                tfd.Gamma(self.emission_diag_factor_conc, self.emission_diag_factor_rate)\
                    .sample(seed=key2, sample_shape=((self.num_states, self.emission_dim)))
            _emission_cov_low_rank_factors = jr.normal(key3, (self.num_states, self.emission_dim, self.emission_rank))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = dict(means=default(emission_means, _emission_means),
                      cov_diag_factors=default(emission_cov_diag_factors, _emission_cov_diag_factors),
                      cov_low_rank_factors=default(emission_cov_low_rank_factors, _emission_cov_low_rank_factors))
        props = dict(means=ParameterProperties(),
                     cov_diag_factors=ParameterProperties(constrainer=tfb.Softplus()),
                     cov_low_rank_factors=ParameterProperties())
        return params, props

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, covariates=None):
        return tfd.MultivariateNormalDiagPlusLowRankCovariance(
            params["means"][state],
            params["cov_diag_factors"][state],
            params["cov_low_rank_factors"][state]
        )

    def log_prior(self, params):
        lp = tfd.Gamma(self.emission_diag_factor_conc, self.emission_diag_factor_rate)\
            .log_prob(params["cov_diag_factors"]).sum()
        return lp


# Now for the models
class GaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = GaussianHMMEmissions(num_states, emission_dim,
                                                  emission_prior_mean=emission_prior_mean,
                                                  emission_prior_concentration=emission_prior_concentration,
                                                  emission_prior_scale=emission_prior_scale,
                                                  emission_prior_extra_df=emission_prior_extra_df)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_means: jnp.array=None,
                   emission_covariances: jnp.array=None,
                   emissions: jnp.array=None,
                   ):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions)
        return params, props


class DiagonalGaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_concentration=1.1,
                 emission_prior_scale=1.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = DiagonalGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_concentration=emission_prior_mean_concentration,
            emission_prior_concentration=emission_prior_concentration,
            emission_prior_scale=emission_prior_scale)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_means: jnp.array=None,
                   emission_scale_diags: jnp.array=None,
                   emissions: jnp.array=None,
                   ):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_scale_diags=emission_scale_diags, emissions=emissions)
        return params, props


class SphericalGaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_mean_covariance=1.0,
                 emission_var_concentration=1.1,
                 emission_var_rate=1.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = SphericalGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_covariance=emission_prior_mean_covariance,
            emission_var_concentration=emission_var_concentration,
            emission_var_rate=emission_var_rate)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_means: jnp.array=None,
                   emission_scales: jnp.array=None,
                   emissions: jnp.array=None,
                   ):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_scales=emission_scales, emissions=emissions)
        return params, props


class SharedCovarianceGaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = SharedCovarianceGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_concentration=emission_prior_concentration,
            emission_prior_scale=emission_prior_scale,
            emission_prior_extra_df=emission_prior_extra_df)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_means: jnp.array=None,
                   emission_covariance: jnp.array=None,
                   emissions: jnp.array=None,
                   ):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariance=emission_covariance, emissions=emissions)
        return params, props


class LowRankGaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 emission_rank: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_diag_factor_concentration=1.1,
                 emission_diag_factor_rate=1.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = LowRankGaussianHMMEmissions(
            num_states, emission_dim, emission_rank,
            emission_diag_factor_concentration=emission_diag_factor_concentration,
            emission_diag_factor_rate=emission_diag_factor_rate)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: jnp.array=None,
                   transition_matrix: jnp.array=None,
                   emission_means=None,
                   emission_cov_diag_factors=None,
                   emission_cov_low_rank_factors=None,
                   emissions: jnp.array=None,
                   ):
        if key is not None:
            key1, key2, key3 = jr.split(key , 3)
        else:
            key1 = key2 = key3 = None

        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_cov_diag_factors=emission_cov_diag_factors, emission_cov_low_rank_factors=emission_cov_low_rank_factors, emissions=emissions)
        return params, props
