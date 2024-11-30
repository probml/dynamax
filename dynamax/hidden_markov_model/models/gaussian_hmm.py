import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jaxtyping import Float, Array
import optax
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import Scalar
from dynamax.utils.distributions import InverseWishart
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart
from dynamax.utils.distributions import nig_posterior_update
from dynamax.utils.distributions import niw_posterior_update
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import pytree_sum
from dynamax.utils.cluster import kmeans_sklearn
from typing import NamedTuple, Optional, Tuple, Union


class ParamsGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


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

    def distribution(self, params, state, inputs=None):
        return tfd.MultivariateNormalFullCovariance(
            params.means[state], params.covs[state])

    def log_prior(self, params):
        return NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params.covs, params.means)).sum()

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   emission_means=None,
                   emission_covariances=None,
                   emissions=None):
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            _emission_means, _ = kmeans_sklearn(self.num_states, emissions.reshape(-1, self.emission_dim), key)
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
        params = ParamsGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsGaussianHMMEmissions(
            means=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        return dict(
            sum_w=jnp.einsum("tk->k", expected_states),
            sum_x=jnp.einsum("tk,ti->ki", expected_states, emissions),
            sum_xxT=jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        )

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.covs.trainable and props.means.trainable:
            niw_prior = NormalInverseWishart(loc=self.emission_prior_mean,
                                            mean_concentration=self.emission_prior_conc,
                                            df=self.emission_prior_df,
                                            scale=self.emission_prior_scale)

            # Find the posterior parameters of the NIW distribution
            def _single_m_step(stats):
                niw_posterior = niw_posterior_update(niw_prior, (stats['sum_x'], stats['sum_xxT'], stats['sum_w']))
                return niw_posterior.mode()

            emission_stats = pytree_sum(batch_stats, axis=0)
            covs, means = vmap(_single_m_step)(emission_stats)
            params = params._replace(means=means, covs=covs)

        elif props.covs.trainable and not props.means.trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed means and trainable covariance")

        elif not props.covs.trainable and props.means.trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed covariance and trainable means")

        return params, m_step_state


class ParamsDiagonalGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    scale_diags: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]


class DiagonalGaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_mean_concentration=1e-4,
                 emission_prior_concentration=0.1,
                 emission_prior_scale=0.1):

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

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            _emission_means, _ = kmeans_sklearn(self.num_states, emissions.reshape(-1, self.emission_dim), key)
            _emission_scale_diags = jnp.ones((self.num_states, self.emission_dim))

        elif method.lower() == "prior":
            this_key, key = jr.split(key)
            prior = NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_conc,
                                       self.emission_prior_conc, self.emission_prior_scale)
            (_emission_vars, _emission_means) = prior.sample(seed=this_key, sample_shape=(self.num_states,))
            _emission_scale_diags = jnp.sqrt(_emission_vars)

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsDiagonalGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            scale_diags=default(emission_scale_diags, _emission_scale_diags))
        props = ParamsDiagonalGaussianHMMEmissions(
            means=ParameterProperties(),
            scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, params, state, inputs=None):
        return tfd.MultivariateNormalDiag(params.means[state],
                                          params.scale_diags[state])

    def log_prior(self, params):
        prior =  NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_conc,
                                    self.emission_prior_conc, self.emission_prior_scale)
        return prior.log_prob((params.scale_diags**2,
                               params.means)).sum()

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xsq = jnp.einsum("tk,ti->ki", expected_states, emissions**2)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_xsq=sum_xsq)

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
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
        scale_diags = jnp.sqrt(vars)
        params = params._replace(means=means, scale_diags=scale_diags)
        return params, m_step_state


class ParamsSphericalGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    scales: Union[Float[Array, "state_dim"], ParameterProperties]


class SphericalGaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_mean_covariance=1.0,
                 emission_var_concentration=1.1,
                 emission_var_rate=1.1,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
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
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_means (array, optional): manually specified emission means.
            emission_scales (array, optional): manually specified emission scales (sqrt of diagonal of spherical covariance matrix).
            emissions (array, optional): emissions for initializing the parameters with kmeans.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            _emission_means, _ = kmeans_sklearn(self.num_states, emissions.reshape(-1, self.emission_dim), key)
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
        params = ParamsSphericalGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            scales=default(emission_scales, _emission_scales))
        props = ParamsSphericalGaussianHMMEmissions(
            means=ParameterProperties(),
            scales=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, params, state, inputs=None):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params.means[state],
                                          params.scales[state] * jnp.ones((dim,)))

    def log_prior(self, params):
        lp = tfd.MultivariateNormalFullCovariance(
            self.emission_prior_mean, self.emission_prior_mean_cov)\
                .log_prob(params.means).sum()
        lp += tfd.Gamma(self.emission_var_concentration, self.emission_var_rate)\
            .log_prob(params.scales**2).sum()
        return lp


class ParamsSharedCovarianceGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    cov: Union[Float[Array, "emission_dim emission_dim"], ParameterProperties]


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
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_means (array, optional): manually specified emission means.
            emission_covariance (array, optional): manually specified emission covariance.
            emissions (array, optional): emissions for initializing the parameters with kmeans.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            _emission_means, _ = kmeans_sklearn(self.num_states, emissions.reshape(-1, self.emission_dim), key)
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
        params = ParamsSharedCovarianceGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            cov=default(emission_covariance, _emission_cov))
        props = ParamsSharedCovarianceGaussianHMMEmissions(
            means=ParameterProperties(),
            cov=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def distribution(self, params, state, inputs=None):
        return tfd.MultivariateNormalFullCovariance(
            params.means[state], params.cov)

    def log_prior(self, params):
        mus = params.means
        Sigma = params.cov
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        lp = InverseWishart(nu0, Psi0).log_prob(Sigma)
        lp += tfd.MultivariateNormalFullCovariance(mu0, Sigma / kappa0).log_prob(mus).sum()
        return lp

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("ti,tj->ij", emissions, emissions)
        sum_T = len(emissions)
        stats = dict(sum_w=sum_w, sum_x=sum_x, sum_xxT=sum_xxT, sum_T=sum_T)
        return stats

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        mu0 = self.emission_prior_mean
        kappa0 = self.emission_prior_conc
        Psi0 = self.emission_prior_scale
        nu0 = self.emission_prior_df

        emission_stats = pytree_sum(batch_stats, axis=0)
        sum_T = emission_stats['sum_T'] + nu0 + self.num_states + self.emission_dim + 1
        sum_w = emission_stats['sum_w'] + kappa0
        sum_x = emission_stats['sum_x'] + kappa0 * mu0
        sum_xxT = emission_stats['sum_xxT'] + Psi0 + kappa0 * jnp.outer(mu0, mu0)
        means = jnp.einsum('ki,k->ki', sum_x, 1/sum_w)
        cov = (sum_xxT - jnp.einsum('ki,kj,k->ij', sum_x, sum_x, 1/sum_w)) / sum_T
        params = params._replace(means=means, cov=cov)
        return params, m_step_state


class ParamsLowRankGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    cov_diag_factors: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    cov_low_rank_factors: Union[Float[Array, "state_dim emission_dim emission_rank"], ParameterProperties]


class LowRankGaussianHMMEmissions(HMMEmissions):

    def __init__(self, num_states, emission_dim, emission_rank,
                 emission_diag_factor_concentration=1.1,
                 emission_diag_factor_rate=1.1,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
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
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_means (array, optional): manually specified emission means.
            emission_cov_diag_factors (array, optional): manually specified diagonals of the emission covariances.
            emission_cov_low_rank_factors (array, optional): manually specified low rank factors of the emission covariances.
            emissions (array, optional): emissions for initializing the parameters with kmeans.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            _emission_means, _ = kmeans_sklearn(self.num_states, emissions.reshape(-1, self.emission_dim), key)
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
        params = ParamsLowRankGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            cov_diag_factors=default(emission_cov_diag_factors, _emission_cov_diag_factors),
            cov_low_rank_factors=default(emission_cov_low_rank_factors, _emission_cov_low_rank_factors))
        props = ParamsLowRankGaussianHMMEmissions(
            means=ParameterProperties(),
            cov_diag_factors=ParameterProperties(constrainer=tfb.Softplus()),
            cov_low_rank_factors=ParameterProperties())
        return params, props

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, inputs=None):
        return tfd.MultivariateNormalDiagPlusLowRankCovariance(
            params.means[state],
            params.cov_diag_factors[state],
            params.cov_low_rank_factors[state]
        )

    def log_prior(self, params):
        lp = tfd.Gamma(self.emission_diag_factor_conc, self.emission_diag_factor_rate)\
            .log_prob(params.cov_diag_factors).sum()
        return lp


### Now for the models ###
class ParamsGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsGaussianHMMEmissions


class GaussianHMM(HMM):
    r"""An HMM with multivariate normal (i.e. Gaussian) emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \Sigma_{z_t})$$

    with $\theta = \{\mu_k, \Sigma_k\}_{k=1}^K$ denoting the *emission means* and *emission covariances*.

    The model has a conjugate normal-inverse-Wishart_ prior,

    $$p(\theta) = \prod_{k=1}^K \mathcal{N}(\mu_k \mid \mu_0, \kappa_0^{-1} \Sigma_k) \mathrm{IW}(\Sigma_{k} \mid \nu_0, \Psi_0)$$

    .. _normal-inverse-Wishart: https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_concentration: $\kappa_0$
    :param emission_prior_extra_df: $\nu_0 - N > 0$, the "extra" degrees of freedom, above and beyond the minimum of $\\nu_0 = N$.
    :param emission_prior_scale: $\Psi_0$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, "emission_dim"]]=0.0,
                 emission_prior_concentration: Scalar=1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]]=1e-4,
                 emission_prior_extra_df: Scalar=0.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = GaussianHMMEmissions(num_states, emission_dim,
                                                  emission_prior_mean=emission_prior_mean,
                                                  emission_prior_concentration=emission_prior_concentration,
                                                  emission_prior_scale=emission_prior_scale,
                                                  emission_prior_extra_df=emission_prior_extra_df)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsGaussianHMM(**params), ParamsGaussianHMM(**props)


class ParamsDiagonalGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsDiagonalGaussianHMMEmissions


class DiagonalGaussianHMM(HMM):
    r"""An HMM with conditionally independent normal (i.e. Gaussian) emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathcal{N}(y_{t,n} \mid \mu_{z_t,n}, \sigma_{z_t,n}^2)$$
    or equivalently
    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \mathrm{diag}(\sigma_{z_t}^2))$$


    where $\sigma_k^2 = [\sigma_{k,1}^2, \ldots, \sigma_{k,N}^2]$ are the *emission variances* of each
    dimension in state $z_t=k$. The complete set of parameters is $\theta = \{\mu_k, \sigma_k^2\}_{k=1}^K$.

    The model has a conjugate normal-inverse-gamma_ prior,

    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathcal{N}(\mu_{k,n} \mid \mu_0, \kappa_0^{-1} \sigma_{k,n}^2) \mathrm{IGa}(\sigma_{k,n}^2 \mid \alpha_0, \beta_0)$$

    .. _normal-inverse-gamma: https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_mean_concentration: $\kappa_0$
    :param emission_prior_concentration: $\alpha_0$
    :param emission_prior_scale: $\\beta_0$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, "emission_dim"]]=0.0,
                 emission_prior_mean_concentration: Union[Scalar, Float[Array, "emission_dim"]]=1e-4,
                 emission_prior_concentration: Scalar=0.1,
                 emission_prior_scale: Scalar=0.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = DiagonalGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_concentration=emission_prior_mean_concentration,
            emission_prior_concentration=emission_prior_concentration,
            emission_prior_scale=emission_prior_scale)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_scale_diags: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_scale_diags: manually specified emission standard deviations $\sigma_{k,n}$
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_scale_diags=emission_scale_diags, emissions=emissions)
        return ParamsDiagonalGaussianHMM(**params), ParamsDiagonalGaussianHMM(**props)


class ParamsSphericalGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsSphericalGaussianHMMEmissions


class SphericalGaussianHMM(HMM):
    r"""An HMM with conditionally independent normal emissions with the same variance along
    each dimension. These are called *spherical* Gaussian emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathcal{N}(y_{t,n} \mid \mu_{z_t,n}, \sigma_{z_t}^2)$$
    or equivalently
    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \sigma_{z_t}^2 I)$$

    where $\sigma_k^2$ is the *emission variance* in state $z_t=k$.
    The complete set of parameters is $\theta = \{\mu_k, \sigma_k^2\}_{k=1}^K$.

    The model has a non-conjugate, factored prior

    $$p(\theta) = \prod_{k=1}^K \mathcal{N}(\mu_{k} \mid \mu_0, \Sigma_0) \mathrm{Ga}(\sigma_{k}^2 \mid \alpha_0, \beta_0)$$

    *Note: In future versions we may implement a conjugate prior for this model.*

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_mean_covariance: $\Sigma_0$
    :param emission_var_concentration: $\alpha_0$
    :param emission_var_rate: $\beta_0$
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, "emission_dim"]]=0.0,
                 emission_prior_mean_covariance: Union[Scalar, Float[Array, "emission_dim emission_dim"]]=1.0,
                 emission_var_concentration: Scalar=1.1,
                 emission_var_rate: Scalar=1.1,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = SphericalGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_covariance=emission_prior_mean_covariance,
            emission_var_concentration=emission_var_concentration,
            emission_var_rate=emission_var_rate,
            m_step_optimizer=m_step_optimizer,
            m_step_num_iters=m_step_num_iters)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_scales: Optional[Float[Array, "num_states"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_scales: manually specified emission scales (sqrt of diagonal of covariance matrix).
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_scales=emission_scales, emissions=emissions)
        return ParamsSphericalGaussianHMM(**params), ParamsSphericalGaussianHMM(**props)


class ParamsSharedCovarianceGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsSharedCovarianceGaussianHMMEmissions


class SharedCovarianceGaussianHMM(HMM):
    r"""An HMM with multivariate normal (i.e. Gaussian) emissions where the covariance
    matrix is shared by all discrete states.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \Sigma)$$

    where $\Sigma$ is the *shared emission covariance*.

    The complete set of parameters is $\theta = (\{\mu_k\}_{k=1}^K, \Sigma)$.

    The model has a conjugate prior,

    $$p(\theta) = \mathrm{IW}(\Sigma \mid \nu_0, \Psi_0) \prod_{k=1}^K \mathcal{N}(\mu_{k} \mid \mu_0, \kappa_0^{-1} \Sigma)$$

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_concentration: $\kappa_0$
    :param emission_prior_scale: $\Psi_0$
    :param emission_prior_extra_df: $\nu_0 - N > 0$, the "extra" degrees of freedom, above and beyond the minimum of $\\nu_0 = N$.

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, "emission_dim"]]=0.0,
                 emission_prior_concentration: Scalar=1e-4,
                 emission_prior_scale: Scalar=1e-4,
                 emission_prior_extra_df: Scalar=0.1):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = SharedCovarianceGaussianHMMEmissions(
            num_states, emission_dim,
            emission_prior_mean=emission_prior_mean,
            emission_prior_concentration=emission_prior_concentration,
            emission_prior_scale=emission_prior_scale,
            emission_prior_extra_df=emission_prior_extra_df)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariance:  Optional[Float[Array, "emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_covariance: manually specified emission covariance.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariance=emission_covariance, emissions=emissions)
        return ParamsSharedCovarianceGaussianHMM(**params), ParamsSharedCovarianceGaussianHMM(**props)


class ParamsLowRankGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLowRankGaussianHMMEmissions


class LowRankGaussianHMM(HMM):
    r"""An HMM with multivariate normal (i.e. Gaussian) emissions where the covariance
    matrix is low rank plus diagonal.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \Sigma_{z_t})$$

    where $\Sigma_k$ factors as,

    $$\Sigma_k = U_k U_k^\top + \mathrm{diag}(d_k)$$

    with *low rank factors* $U_k \in \mathbb{R}^{N \times M}$ and
    *diagonal factor* $d_k \in \mathbb{R}_+^{N}$.

    The complete set of parameters is $\theta = (\{\mu_k, U_k, d_k\}_{k=1}^K$.

    This model does not have a conjugate prior. Instead, we place a gamma prior on the diagonal factors,

    $$p(\theta) \propto \prod_{k=1}^K \prod_{n=1}^N \mathrm{Ga}(d_{k,n} \mid \alpha_0, \beta_0)$$

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param emission_rank: rank of the low rank factors, $M$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_diag_factor_concentration: $\alpha_0$
    :param emission_diag_factor_rate: $\beta_0$
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 emission_rank: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_diag_factor_concentration: Scalar=1.1,
                 emission_diag_factor_rate: Scalar=1.1,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):

        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LowRankGaussianHMMEmissions(
            num_states, emission_dim, emission_rank,
            emission_diag_factor_concentration=emission_diag_factor_concentration,
            emission_diag_factor_rate=emission_diag_factor_rate,
            m_step_optimizer=m_step_optimizer,
            m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_cov_diag_factors: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_cov_low_rank_factors: Optional[Float[Array, "num_states emission_dim emission_rank"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_cov_diag_factors: manually specified emission scales (sqrt of diagonal of covariance matrix).
            emission_cov_low_rank_factors: manually specified emission low rank factors (sqrt of diagonal of covariance matrix).
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_cov_diag_factors=emission_cov_diag_factors, emission_cov_low_rank_factors=emission_cov_low_rank_factors, emissions=emissions)
        return ParamsLowRankGaussianHMM(**params), ParamsLowRankGaussianHMM(**props)
