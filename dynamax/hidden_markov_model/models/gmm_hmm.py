"""
Gaussian Mixture Hidden Markov Models (GMM-HMMs) and Diagonal Gaussian Mixture Hidden Markov Models (Diag-GMM-HMMs).
"""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.scipy.special import logsumexp
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart
from dynamax.utils.distributions import nig_posterior_update
from dynamax.utils.distributions import niw_posterior_update
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import pytree_sum
from dynamax.types import IntScalar, Scalar


class ParamsGaussianMixtureHMMEmissions(NamedTuple):
    """Parameters for a Gaussian Mixture HMM emission distribution."""
    weights: Union[Float[Array, "state_dim num_components"], ParameterProperties]
    means: Union[Float[Array, "state_dim num_components emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim num_components emission_dim emission_dim"], ParameterProperties]


class ParamsGaussianMixtureHMM(NamedTuple):
    """Parameters for a Gaussian Mixture HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsGaussianMixtureHMMEmissions


class ParamsDiagonalGaussianMixtureHMMEmissions(NamedTuple):
    """Parameters for a Diagonal Gaussian Mixture HMM emission distribution."""
    weights: Union[Float[Array, "state_dim num_components"], ParameterProperties]
    means: Union[Float[Array, "state_dim num_components emission_dim"], ParameterProperties]
    scale_diags: Union[Float[Array, "state_dim num_components emission_dim"], ParameterProperties]


class ParamsDiagonalGaussianMixtureHMM(NamedTuple):
    """Parameters for a Diagonal Gaussian Mixture HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsDiagonalGaussianMixtureHMMEmissions


# Emissions and models
class GaussianMixtureHMMEmissions(HMMEmissions):
    r"""Emission distribution for a Gaussian Mixture HMM.

    The emission distribution is a mixture of multivariate normal (i.e. Gaussian) distributions.

    Args:
        num_states: number of discrete states $K$
        num_components: number of mixture components $C$
        emission_dim: number of conditionally independent emissions $N$
        emission_weights_concentration: $\gamma$
        emission_prior_mean: $\mu_0$
        emission_prior_mean_concentration: $\kappa_0$
        emission_prior_extra_df: $\nu_0 - N > 0$, the "extra" degrees of freedom, above and beyond the minimum of $\nu_0 = N$.
        emission_prior_scale: $\Psi_
    """
    def __init__(self,
                 num_states: int,
                 num_components: int,
                 emission_dim: int,
                 emission_weights_concentration: Union[Scalar, Float[Array, " num_components"]]=1.1,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]]=0.,
                 emission_prior_mean_concentration: Scalar=1e-4,
                 emission_prior_extra_df: Scalar=1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]]=0.1):
        self.num_states = num_states
        self.num_components = num_components
        self.emission_dim = emission_dim
        self.emission_weights_concentration = emission_weights_concentration * jnp.ones(num_components)
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_mean_concentration = emission_prior_mean_concentration
        self.emission_prior_df = emission_dim + emission_prior_extra_df
        self.emission_prior_scale = emission_prior_scale * jnp.eye(emission_dim)

    @property
    def emission_shape(self):
        """Shape of the emission distribution."""
        return (self.emission_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   emission_weights: Optional[Float[Array, "num_states num_components"]]=None,
                   emission_means: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
                   emission_covariances: Optional[Float[Array, "num_states num_components emission_dim emission_dim"]]=None,
                   emissions: Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[ParamsGaussianMixtureHMMEmissions, ParamsGaussianMixtureHMMEmissions]:
        """
        Initialize the emission parameters.
        
        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_weights: manually specified emission weights.
            emission_means: manually specified emission means.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.ones((self.num_states, self.num_components)) / self.num_components
            _emission_means = jnp.tile(jnp.array(km.cluster_centers_)[:, None, :], (1, self.num_components, 1))
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, self.num_components, 1, 1))

        elif method.lower() == "prior":
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jr.dirichlet(key1, self.emission_weights_concentration, shape=(self.num_states,))

            prior = NormalInverseWishart(self.emission_prior_mean,
                                         self.emission_prior_mean_concentration,
                                         self.emission_prior_df,
                                         self.emission_prior_scale)
            (_emission_covs, _emission_means) = prior.sample(
                seed=key2, sample_shape=(self.num_states, self.num_components))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsGaussianMixtureHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            means=default(emission_means, _emission_means),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsGaussianMixtureHMMEmissions(
            weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            means=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def distribution(self, 
                     params: ParamsGaussianMixtureHMMEmissions, 
                     state: IntScalar,
                     inputs: Optional[Array] = None
        ) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=params.weights[state]),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=params.means[state], covariance_matrix=params.covs[state]))

    def log_prior(self, params:ParamsGaussianMixtureHMMEmissions) -> Float[Array, ""]:
        """Compute the log prior probability of the emission parameters."""
        lp = tfd.Dirichlet(self.emission_weights_concentration).log_prob(
            params.weights).sum()
        lp += NormalInverseWishart(self.emission_prior_mean, self.emission_prior_mean_concentration,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params.covs, params.means)).sum()
        return lp

    def collect_suff_stats(self, 
                           params: ParamsGaussianMixtureHMMEmissions, 
                           posterior: HMMPosterior,
                           emissions: Float[Array, "num_timesteps emission_dim"], 
                           inputs: Optional[Array] = None
        ) -> Dict[str, Float[Array, "..."]]:
        """Collect sufficient statistics for the emission parameters."""
        
        def prob_fn(x):
            """Compute the posterior probability of each discrete class."""
            logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalFullCovariance(
                loc=mus, covariance_matrix=sigmas).log_prob(x) + jnp.log(weights))(
                    params.means, params.covs, params.weights)
            logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
            return jnp.exp(logprobs)

        prob_denses = vmap(prob_fn)(emissions)
        expected_states = posterior.smoothed_probs
        weights = jnp.einsum("tk,tkm->tkm", expected_states, prob_denses)
        Sx = jnp.einsum("tkm,tn->kmn", weights, emissions)
        SxxT = jnp.einsum("tkm,tn,tl->kmnl", weights, emissions, emissions)
        N = weights.sum(axis=0)
        return dict(N=N, Sx=Sx, SxxT=SxxT)

    def initialize_m_step_state(
            self,
            params: ParamsGaussianMixtureHMMEmissions,
            props: ParamsGaussianMixtureHMMEmissions
        ) -> None:
        """Initialize the M-step state."""
        return None

    def m_step(
            self,
            params: ParamsGaussianMixtureHMMEmissions,
            props: ParamsGaussianMixtureHMMEmissions,
            batch_stats: Dict[str, Float[Array, "..."]],
            m_step_state: Any
    ) -> Tuple[ParamsGaussianMixtureHMMEmissions, Any]:
        """Perform the M-step of the EM algorithm."""
        assert props.weights.trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"
        assert props.means.trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"
        assert props.covs.trainable, "GaussianMixtureHMM.fit_em() does not support fitting a subset of parameters"

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

        emission_stats = pytree_sum(batch_stats, axis=0)
        weights, means, covs = vmap(_single_m_step)(
            emission_stats['Sx'], emission_stats['SxxT'], emission_stats['N'])
        params = params._replace(weights=weights, means=means, covs=covs)
        return params, m_step_state


class GaussianMixtureHMM(HMM):
    r"""An HMM with mixture of multivariate normal (i.e. Gaussian) emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \sum_{c=1}^C w_{k,c} \mathcal{N}(y_{t} \mid \mu_{z_t, c}, \Sigma_{z_t, c})$$

    with $\theta = \{\{\mu_{k,c}, \Sigma_{k, c}\}_{c=1}^C, w_k \}_{k=1}^K$ denoting
    the *emission means*  and *emission covariances* for each disrete state $k$ and *component* $c$,
    as well as the *emission weights* $w_k \in \Delta_C$, which specify the probability of each
    component in state $k$.

    The model has a conjugate normal-inverse-Wishart_ prior,

    $$p(\theta) = \mathrm{Dir}(w_k \mid \gamma 1_C) \prod_{k=1}^K \prod_{c=1}^C \mathcal{N}(\mu_{k,c} \mid \mu_0, \kappa_0^{-1} \Sigma_{k,c}) \mathrm{IW}(\Sigma_{k, c} \mid \nu_0, \Psi_0)$$

    .. _normal-inverse-Wishart: https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    :param num_states: number of discrete states $K$
    :param num_components: number of mixture components $C$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_weights_concentration=: $\gamma$
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_concentration: $\kappa_0$
    :param emission_prior_extra_df: $\nu_0 - N > 0$, the "extra" degrees of freedom, above and beyond the minimum of $\nu_0 = N$.
    :param emission_prior_scale: $\Psi_0$

    """
    def __init__(self,
                 num_states: int,
                 num_components: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_weights_concentration: Union[Scalar, Float[Array, " num_components"]]=1.1,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]]=0.0,
                 emission_prior_mean_concentration: Scalar=1e-4,
                 emission_prior_extra_df: Scalar=1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]]=1e-4):
        self.emission_dim = emission_dim
        self.num_components = num_components
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = GaussianMixtureHMMEmissions(
            num_states, num_components, emission_dim,
            emission_weights_concentration=emission_weights_concentration,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_concentration=emission_prior_mean_concentration,
            emission_prior_scale=emission_prior_scale,
            emission_prior_extra_df=emission_prior_extra_df)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states num_components"]]=None,
                   emission_means: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states num_components emission_dim emission_dim"]]=None,
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
            emission_weights: manually specified emission weights.
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
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsGaussianMixtureHMM(**params), ParamsGaussianMixtureHMM(**props)


class DiagonalGaussianMixtureHMMEmissions(HMMEmissions):
    r"""Emission distribution for a Diagonal Gaussian Mixture HMM.

    The emission distribution is a mixture of multivariate normal (i.e. Gaussian) 
    distributions with diagonal covariance.

    Args:
        num_states: number of discrete states $K$
        num_components: number of mixture components $C$
        emission_dim: number of conditionally independent emissions $N$
        emission_weights_concentration: $\gamma$
        emission_prior_mean: $\mu_0$
        emission_prior_mean_concentration: $\kappa_0$
        emission_prior_shape: $\alpha_0$
        emission_prior_scale: $\beta_0$
    """
    def __init__(self,
                 num_states: int,
                 num_components: int,
                 emission_dim: int,
                 emission_weights_concentration: Union[Scalar, Float[Array, " num_components"]]=1.1,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]]=0.,
                 emission_prior_mean_concentration: Scalar=1e-4,
                 emission_prior_shape: Scalar=1.,
                 emission_prior_scale: Union[Scalar, Float[Array, " emission_dim"]]=1.):
        self.num_states = num_states
        self.num_components = num_components
        self.emission_dim = emission_dim

        self.emission_weights_concentration = \
            emission_weights_concentration * jnp.ones(num_components)
        self.emission_prior_mean = emission_prior_mean
        self.emission_prior_mean_concentration = emission_prior_mean_concentration
        self.emission_prior_shape = emission_prior_shape
        self.emission_prior_scale = emission_prior_scale

    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of the emission distribution."""
        return (self.emission_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   emission_weights: Optional[Float[Array, "num_states num_components"]]=None,
                   emission_means: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
                   emission_scale_diags: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
                   emissions: Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[ParamsDiagonalGaussianMixtureHMMEmissions, ParamsDiagonalGaussianMixtureHMMEmissions]:
        """
        Initialize the emission parameters.
        
        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_weights: manually specified emission weights.
            emission_means: manually specified emission means.
            emission_scale_diags: manually specified emission scale diagonals.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.ones((self.num_states, self.num_components)) / self.num_components
            _emission_means = jnp.tile(jnp.array(km.cluster_centers_)[:, None, :], (1, self.num_components, 1))
            _emission_scale_diags = jnp.ones((self.num_states, self.num_components, self.emission_dim))

        elif method.lower() == "prior":
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jr.dirichlet(key1, self.emission_weights_concentration, shape=(self.num_states,))

            prior = NormalInverseGamma(self.emission_prior_mean,
                                       self.emission_prior_mean_concentration,
                                       self.emission_prior_shape,
                                       self.emission_prior_scale)
            (_emission_scale_diags, _emission_means) = prior.sample(
                seed=key2, sample_shape=(self.num_states, self.num_components, self.emission_dim))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsDiagonalGaussianMixtureHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            means=default(emission_means, _emission_means),
            scale_diags=default(emission_scale_diags, _emission_scale_diags))
        props = ParamsDiagonalGaussianMixtureHMMEmissions(
            weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            means=ParameterProperties(),
            scale_diags=ParameterProperties(constrainer=tfb.Softplus()))
        return params, props

    def distribution(self, 
                     params: ParamsDiagonalGaussianMixtureHMMEmissions, 
                     state: IntScalar,
                     inputs: Optional[Array] = None
        ) -> tfd.Distribution:
        """Return the emission distribution for a given state."""

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=params.weights[state]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=params.means[state],
                scale_diag=params.scale_diags[state]))

    def log_prior(self, params: ParamsDiagonalGaussianMixtureHMMEmissions) -> Float[Array, ""]:
        """Compute the log prior probability of the emission parameters."""
        lp = tfd.Dirichlet(self.emission_weights_concentration).log_prob(
            params.weights).sum()
        lp += NormalInverseGamma(self.emission_prior_mean, self.emission_prior_mean_concentration,
                                   self.emission_prior_shape, self.emission_prior_scale).log_prob(
            (params.scale_diags**2, params.means)).sum()
        return lp

    # Expectation-maximization (EM) code
    def collect_suff_stats(self, 
                           params: ParamsDiagonalGaussianMixtureHMMEmissions, 
                           posterior: HMMPosterior,
                           emissions: Float[Array, "num_timesteps emission_dim"], 
                           inputs: Optional[Array] = None
        ) -> Dict[str, Float[Array, "..."]]:
        """Collect sufficient statistics for the emission parameters."""

        # Evaluate the posterior probability of each discrete class
        def prob_fn(x):
            """Compute the posterior probability of each discrete class."""
            logprobs = vmap(lambda mus, sigmas, weights: tfd.MultivariateNormalDiag(
                loc=mus, scale_diag=sigmas).log_prob(x) + jnp.log(weights))(
                    params.means, params.scale_diags,
                    params.weights)
            logprobs = logprobs - logsumexp(logprobs, axis=-1, keepdims=True)
            return jnp.exp(logprobs)

        prob_denses = vmap(prob_fn)(emissions)
        expected_states = posterior.smoothed_probs
        weights = jnp.einsum("tk,tkm->tkm", expected_states, prob_denses)
        Sx = jnp.einsum("tkm,tn->kmn", weights, emissions)
        Sxsq = jnp.einsum("tkm,tn,tn->kmn", weights, emissions, emissions)
        N = weights.sum(axis=0)
        return dict(N=N, Sx=Sx, Sxsq=Sxsq)

    def initialize_m_step_state(self, 
                                params: ParamsDiagonalGaussianMixtureHMMEmissions, 
                                props: ParamsDiagonalGaussianMixtureHMMEmissions
    ) -> None:
        """Initialize the M-step state."""
        return None

    def m_step(self, 
               params: ParamsDiagonalGaussianMixtureHMMEmissions, 
               props: ParamsDiagonalGaussianMixtureHMMEmissions, 
               batch_stats: Dict[str, Float[Array, "..."]], 
               m_step_state: None
    ) -> Tuple[ParamsDiagonalGaussianMixtureHMMEmissions, None]:
        """Perform the M-step of the EM algorithm."""
        assert props.weights.trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"
        assert props.means.trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"
        assert props.scale_diags.trainable, "GaussianMixtureDiagHMM.fit_em() does not support fitting a subset of parameters"

        nig_prior = NormalInverseGamma(
            self.emission_prior_mean, self.emission_prior_mean_concentration,
            self.emission_prior_shape, self.emission_prior_scale)

        def _single_m_step(Sx, Sxsq, N):
            """Update the parameters for one discrete state"""
            # Update the component probabilities (i.e. weights)
            nu_post = self.emission_weights_concentration + N
            mixture_weights = tfd.Dirichlet(nu_post).mode()

            # Update the mean and variances for each component
            var_diags, means = vmap(lambda stats: nig_posterior_update(nig_prior, stats).mode())((Sx, Sxsq, N))
            scale_diags = jnp.sqrt(var_diags)
            return mixture_weights, means, scale_diags

        # Compute mixture weights, diagonal factors of covariance matrices and means
        # for each state in parallel. Note that the first dimension of all sufficient
        # statistics is equal to number of states of HMM.
        emission_stats = pytree_sum(batch_stats, axis=0)
        weights, means, scale_diags = vmap(_single_m_step)(
            emission_stats['Sx'], emission_stats['Sxsq'], emission_stats['N'])
        params = params._replace(weights=weights, means=means, scale_diags=scale_diags)
        return params, m_step_state


class DiagonalGaussianMixtureHMM(HMM):
    r"""An HMM with mixture of multivariate normal (i.e. Gaussian) emissions with diagonal covariance.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \sum_{c=1}^C w_{k,c} \mathcal{N}(y_{t} \mid \mu_{z_t, c}, \mathrm{diag}(\sigma_{z_t, c}^2))$$

    or, equivalently,

    $$p(y_t \mid z_t, \theta) = \sum_{c=1}^C w_{k,c} \prod_{n=1}^N \mathcal{N}(y_{t,n} \mid \mu_{z_t, c, n}, \sigma_{z_t, c, n}^2)$$

    The parameters are $\theta = \{\{\mu_{k,c}, \sigma_{k, c}^2\}_{c=1}^C, w_k \}_{k=1}^K$ denoting
    the *emission means*  and *emission variances* for each disrete state $k$ and *component* $c$,
    as well as the *emission weights* $w_k \in \Delta_C$, which specify the probability of each
    component in state $k$.

    The model has a conjugate normal-inverse-gamma_ prior,

    $$p(\theta) = \mathrm{Dir}(w_k \mid \gamma 1_C) \prod_{k=1}^K \prod_{c=1}^C \prod_{n=1}^N \mathcal{N}(\mu_{k,c,n} \mid \mu_0, \kappa_0^{-1} \sigma_{k,c}^2) \mathrm{IGa}(\sigma_{k, c, n}^2 \mid \alpha_0, \beta_0)$$

    .. _normal-inverse-gamma: https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution

    :param num_states: number of discrete states $K$
    :param num_components: number of mixture components $C$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_weights_concentration=: $\gamma$
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_mean_concentration: $\kappa_0$
    :param emission_prior_shape: $\alpha_0$
    :param emission_prior_scale: $\beta_0$

    """
    def __init__(self,
                 num_states: int,
                 num_components: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_weights_concentration: Union[Scalar, Float[Array, " num_components"]]=1.1,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]]=0.0,
                 emission_prior_mean_concentration: Scalar=1e-4,
                 emission_prior_shape: Scalar=1.,
                 emission_prior_scale: Scalar=1.):
        self.emission_dim = emission_dim
        self.num_components = num_components
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = DiagonalGaussianMixtureHMMEmissions(
            num_states, num_components, emission_dim,
            emission_weights_concentration=emission_weights_concentration,
            emission_prior_mean=emission_prior_mean,
            emission_prior_mean_concentration=emission_prior_mean_concentration,
            emission_prior_shape=emission_prior_shape,
            emission_prior_scale=emission_prior_scale)
        super().__init__(num_states, initial_component, transition_component, emission_component)


    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states num_components"]]=None,
                   emission_means: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
                   emission_scale_diags: Optional[Float[Array, "num_states num_components emission_dim"]]=None,
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
            emission_weights: manually specified emission weights.
            emission_means: manually specified emission means.
            emission_scale_diags: manually specified emission scales (sqrt of the variances). Defaults to None.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_means=emission_means, emission_scale_diags=emission_scale_diags, emissions=emissions)
        return ParamsDiagonalGaussianMixtureHMM(**params), ParamsDiagonalGaussianMixtureHMM(**props)
