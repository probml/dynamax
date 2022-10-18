from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax import vmap, jit
from dynamax.distributions import InverseWishart as IW
from dynamax.distributions import MatrixNormalPrecision as MN
from dynamax.structural_time_series.models.structural_time_series_ssm import GaussianSSM, PoissonSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


def _set_prior(input_prior, default_prior):
    return input_prior if input_prior is not None else default_prior


class StructuralTimeSeries():
    """The class of the Bayesian structural time series (STS) model

    The STS model is defined by a pair of equations relating
    a time series observation y_t to
    a vector of latent state z_t:

    y_t     =
    z_{t+1} = F_t @ z_t + N(0, Q_t)

    H_t: fixed emission matrix
    D_t: random matrix of regression coefficients
    F_t: fixed dynamics matrix
    R_t: random covariance matrix of the observation noise
    Q_t: random covariance matrix of the latent state

    Construct a structural time series (STS) model from a list of components

    Args:
        components: list of components
        observation_covariance:
        observation_covariance_prior: InverseWishart prior for the observation covariance matrix
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        name (str): name of the STS model
    """

    def __init__(self,
                 components,
                 observed_time_series,
                 observation_distribution_family='Gaussian',
                 observation_covariance=None,
                 observation_covariance_prior=None,
                 name='StructuralTimeSeries'):

        assert observation_distribution_family in ['Gaussian', 'Poisson']

        self.dim_obs = 1 if len(observed_time_series.shape) == 1 else observed_time_series.shape[-1]
        obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
        self.obs_family = observation_distribution_family
        self.name = name

        if self.obs_family == 'Gaussian':
            self.observation_covariance_prior = _set_prior(
                observation_covariance_prior,
                IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs))
                )
            if observation_covariance is not None:
                self.observation_covariance = observation_covariance
            else:
                self.observation_covariance = 1e-4*obs_scale**2*jnp.eye(self.dim_obs)

        # Save parameters of the STS model:
        self.initial_state_priors = OrderedDict()

        self.transition_matrices = OrderedDict()
        self.transition_covariances = OrderedDict()
        self.transition_covariance_priors = OrderedDict()
        self.cov_spars_matrices = OrderedDict()

        self.observation_matrices = OrderedDict()

        self.observation_regression_weights = None
        self.observation_regression_weights_prior = None

        # Aggregate components
        for c in components:
            if isinstance(c, STSLatentComponent):
                self.initial_state_priors.update(c.initial_state_prior)

                self.transition_matrices.update(c.transition_matrix)
                self.transition_covariances.update(c.transition_covariance)
                self.transition_covariance_priors.update(c.transition_covariance_prior)
                self.cov_spars_matrices.update(c.cov_spars_matrix)

                self.observation_matrices.update(c.observation_matrix)

            elif isinstance(c, LinearRegression):
                self.observation_regression_weights = c.weights_prior.mode()
                self.observation_regression_weights_prior = c.weights_prior

    def as_ssm(self):
        """Formulate the STS model as a linear Gaussian state space model:

        p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
        p(y_t | z_t) =
        p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})

        F_t, H_t are fixed known matrices,
        the convariance matrices, Q and R, are random variables to be learned,
        the regression coefficient matrix B is also unknown random matrix
        if the STS model includes an regression component
        """
        if self.obs_family == 'Gaussian':
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  self.transition_covariances,
                                  self.transition_covariance_priors,
                                  self.observation_covariance,
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  self.observation_regression_weights,
                                  self.observation_regression_weights_prior
                                  )
        elif self.obs_family == 'Poisson':
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 self.transition_covariances,
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 self.observation_regression_weights,
                                 self.observation_regression_weights_prior
                                 )
        return sts_ssm

    def decompose_by_component(self, observed_time_series, inputs=None,
                               sts_params=None, num_post_samples=100, key=jr.PRNGKey(0)):
        """Decompose the STS model into components and return the means and variances
           of the marginal posterior of each component.

           The marginal posterior of each component is obtained by averaging over
           conditional posteriors of that component using Kalman smoother conditioned
           on the sts_params. Each sts_params is a posterior sample of the STS model
           conditioned on observed_time_series.

           The marginal posterior mean and variance is computed using the formula
           E[X] = E[E[X|Y]],
           Var(Y) = E[Var(X|Y)] + Var[E[X|Y]],
           which holds for any random variables X and Y.

        Args:
            observed_time_series (_type_): _description_
            inputs (_type_, optional): _description_. Defaults to None.
            sts_params (_type_, optional): Posteriror samples of STS parameters.
                If not given, 'num_posterior_samples' of STS parameters will be
                sampled using self.fit_hmc.
            num_post_samples (int, optional): Number of posterior samples of STS
                parameters to be sampled using self.fit_hmc if sts_params=None.

        Returns:
            component_dists: (OrderedDict) each item is a tuple of means and variances
                              of one component.
        """
        component_dists = OrderedDict()

        # Sample parameters from the posterior if parameters is not given
        if sts_params is None:
            sts_ssm = self.as_ssm()
            sts_params = sts_ssm.fit_hmc(key, num_post_samples, observed_time_series, inputs)

        @jit
        def decomp_poisson(sts_param):
            """Decompose one STS model if the observations follow Poisson distributions.
            """
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior)
            return sts_ssm.component_posterior(observed_time_series, inputs)

        @jit
        def decomp_gaussian(sts_param):
            """Decompose one STS model if the observations follow Gaussian distributions.
            """
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior)
            return sts_ssm.component_posterior(observed_time_series, inputs)

        # Obtain the smoothed posterior for each component given the parameters
        if self.obs_family == 'Gaussian':
            component_conditional_pos = vmap(decomp_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            component_conditional_pos = vmap(decomp_poisson)(sts_params)

        # Obtain the marginal posterior
        for c, pos in component_conditional_pos.items():
            mus = pos[0]
            vars = pos[1]
            # Use the formula: E[X] = E[E[X|Y]]
            mu_series = mus.mean(axis=0)
            # Use the formula: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
            var_series = jnp.mean(vars, axis=0)[..., 0] + jnp.var(mus, axis=0)
            component_dists[c] = (mu_series, var_series)

        return component_dists

    def sample(self, key, num_timesteps, inputs=None):
        """Given parameters, sample latent states and corresponding observed time series.
        """
        sts_ssm = self.as_ssm()
        states, timeseries = sts_ssm.sample(key, num_timesteps, inputs)
        return timeseries

    def marginal_log_prob(self, observed_time_series, inputs=None):
        sts_ssm = self.as_ssm()
        return sts_ssm.marginal_log_prob(observed_time_series, inputs)

    def posterior_sample(self, key, observed_time_series, sts_params, inputs=None):
        @jit
        def single_sample_poisson(sts_param):
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior
                                 )
            ts_means, ts = sts_ssm.posterior_sample(key, observed_time_series, inputs)
            return [ts_means, ts]

        @jit
        def single_sample_gaussian(sts_param):
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior
                                  )
            ts_means, ts = sts_ssm.posterior_sample(key, observed_time_series, inputs)
            return [ts_means, ts]

        if self.obs_family == 'Gaussian':
            samples = vmap(single_sample_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            samples = vmap(single_sample_poisson)(sts_params)

        return {'means': samples[0], 'observations': samples[1]}

    def fit_hmc(self, key, sample_size, observed_time_series, inputs=None,
                warmup_steps=500, num_integration_steps=30):
        """Sample parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            regression coefficient matrix (if the model has inputs and a regression component)
            covariance matrix of observations (if observations follow Gaussian distribution)
        """
        sts_ssm = self.as_ssm()
        param_samps = sts_ssm.fit_hmc(key, sample_size, observed_time_series, inputs,
                                      warmup_steps, num_integration_steps)
        return param_samps

    def fit_vi(self, key, sample_size, observed_time_series, inputs=None, M=100):
        """Sample parameters of the STS model from the approximate distribution fitted by ADVI.
        """
        sts_ssm = self.as_ssm()
        param_samps = sts_ssm.fit_vi(key, sample_size, observed_time_series, inputs, M)
        return param_samps

    def forecast(self, key, observed_time_series, sts_params, num_forecast_steps,
                 past_inputs=None, forecast_inputs=None):
        @jit
        def single_forecast_gaussian(sts_param):
            sts_ssm = GaussianSSM(self.transition_matrices,
                                  self.observation_matrices,
                                  self.initial_state_priors,
                                  sts_param['dynamics_covariances'],
                                  self.transition_covariance_priors,
                                  sts_param['emission_covariance'],
                                  self.observation_covariance_prior,
                                  self.cov_spars_matrices,
                                  sts_param['regression_weights'],
                                  self.observation_regression_weights_prior
                                  )
            means, covs, ts = sts_ssm.forecast(key, observed_time_series, num_forecast_steps,
                                               past_inputs, forecast_inputs)
            return [means, covs, ts]

        @jit
        def single_forecast_poisson(sts_param):
            sts_ssm = PoissonSSM(self.transition_matrices,
                                 self.observation_matrices,
                                 self.initial_state_priors,
                                 sts_param['dynamics_covariances'],
                                 self.transition_covariance_priors,
                                 self.cov_spars_matrices,
                                 sts_param['regression_weights'],
                                 self.observation_regression_weights_prior
                                 )
            means, covs, ts = sts_ssm.forecast(key, observed_time_series, num_forecast_steps,
                                               past_inputs, forecast_inputs)
            return [means, covs, ts]

        if self.obs_family == 'Gaussian':
            forecasts = vmap(single_forecast_gaussian)(sts_params)
        elif self.obs_family == 'Poisson':
            forecasts = vmap(single_forecast_poisson)(sts_params)

        return {'means': forecasts[0], 'covariances': forecasts[1], 'observations': forecasts[2]}


######################################
#    Classes of components of STS    #
######################################


class STSLatentComponent(ABC):

    @property
    @abstractmethod
    def transition_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_covariance(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_covariance_prior(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state_prior(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_spars_matrix(self):
        raise NotImplementedError


class LocalLinearTrend(STSLatentComponent):
    """The local linear trend component of the structual time series (STS) model

    level[t+1] = level[t] + slope[t] + N(0, level_covariance)
    slope[t+1] = slope[t] + N(0, slope_covariance)

    The latent state is [level, slope].

    Args:
        level_covariance_prior: A tfd.Distribution instance, an InverseWishart prior by default
        slope_covariance_prior: A tfd.Distribution instance, an InverseWishart prior by default
        initial_level_prior: A tfd.Distribution prior for the level part of the initial state,
                             a MultivariateNormal by default
        initial_slope_prior: A tfd.Distribution prior for the slope part of the initial state,
                             a MultivariateNormal by default
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        dim_observed_time_series: dimension of the observed time series
        name (str):               name of the component in the STS model
    """

    def __init__(self,
                 level_covariance_prior=None,
                 slope_covariance_prior=None,
                 initial_level_prior=None,
                 initial_slope_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='LocalLinearTrend'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
            obs_init = observed_time_series[0].mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.
            obs_init = 0.

        self.component_name = name

        # Initialize the prior using the observed time series if a prior is not specified
        self.level_covariance_prior = _set_prior(
            level_covariance_prior,
            IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs)))

        self.slope_covariance_prior = _set_prior(
            slope_covariance_prior,
            IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs)))

        self.initial_level_prior = _set_prior(
            initial_level_prior,
            MVN(loc=obs_init * jnp.ones(self.dim_obs),
                covariance_matrix=obs_scale*jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_level_prior, MVN)

        self.initial_slope_prior = _set_prior(
            initial_slope_prior,
            MVN(loc=jnp.zeros(self.dim_obs), covariance_matrix=jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_slope_prior, MVN)

    @property
    def transition_matrix(self):
        return {self.component_name:
                jnp.block([[jnp.eye(self.dim_obs), jnp.eye(self.dim_obs)],
                           [jnp.zeros((self.dim_obs, self.dim_obs)), jnp.eye(self.dim_obs)]])}

    @property
    def observation_matrix(self):
        return {self.component_name:
                jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))])}

    @property
    def transition_covariance(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior.mode(),
                            'local_linear_slope': self.slope_covariance_prior.mode()})

    @property
    def transition_covariance_prior(self):
        return OrderedDict({'local_linear_level': self.level_covariance_prior,
                            'local_linear_slope': self.slope_covariance_prior})

    @property
    def initial_state_prior(self):
        return OrderedDict({'local_linear_level': self.initial_level_prior,
                            'local_linear_slope': self.initial_slope_prior})

    @property
    def cov_spars_matrix(self):
        return OrderedDict({'local_linear_level': jnp.eye(self.dim_obs),
                            'local_linear_slope': jnp.eye(self.dim_obs)})


class LinearRegression():
    """The static regression component of the structual time series (STS) model

    Args:
        weights_prior: MatrixNormal prior for the weight matrix
        weights_shape: Dimension of the observed time series
        name (str): Name of the component in the STS model
    """

    def __init__(self,
                 weights_shape,
                 weights_prior=None,
                 name='LinearRegression'):
        self.dim_obs, self.dim_inputs = weights_shape
        self.component_name = name

        # Initialize the prior distribution for weights
        if weights_prior is None:
            weights_prior = MN(loc=jnp.zeros(weights_shape),
                               row_covariance=jnp.eye(self.dim_obs),
                               col_precision=jnp.eye(self.dim_inputs))

        self.weights_prior = weights_prior


class Seasonal(STSLatentComponent):
    """The seasonal component of the structual time series (STS) model
    Since on average sum_{j=0}^{num_seasons-1}s_{t+1-j} = 0 for any t,
    the seasonal effect (random) for next time step is:

    s_{t+1} = - sum_{j=1}^{num_seasons-1} s_{t+1-j} + N(0, drift_covariance)

    Args:
        num_seasons (int): number of seasons (assuming number of steps per season is 1)
        num_steps_per_season:
        drift_covariance_prior: InverseWishart prior for drift_covariance
        initial_effect_prior: MultivariateNormal prior for initial_effect
        observed_time_series: has shape (batch_size, timesteps, dim_observed_timeseries)
        dim_observed_time_series: dimension of the observed time series
        name (str): name of the component in the STS model
    """

    def __init__(self,
                 num_seasons,
                 num_steps_per_season=1,
                 drift_covariance_prior=None,
                 initial_effect_prior=None,
                 observed_time_series=None,
                 dim_observed_timeseries=1,
                 name='Seasonal'):
        if observed_time_series is not None:
            _dim = observed_time_series.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(jnp.abs(jnp.diff(observed_time_series, axis=0)), axis=0).mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.

        self.num_seasons = num_seasons
        self.num_steps_per_season = num_steps_per_season
        self.component_name = name

        self.initial_effect_prior = _set_prior(
            initial_effect_prior,
            MVN(loc=jnp.zeros(self.dim_obs),
                covariance_matrix=obs_scale**2*jnp.eye(self.dim_obs)))

        self.drift_covariance_prior = _set_prior(
            drift_covariance_prior,
            IW(df=self.dim_obs, scale=1e-4*obs_scale**2*jnp.eye(self.dim_obs)))

    @property
    def transition_matrix(self):
        # TODO: allow num_steps_per_season > 1 or be a list of integers
        return {self.component_name:
                jnp.block([[jnp.kron(-jnp.ones(self.num_seasons-1), jnp.eye(self.dim_obs))],
                           [jnp.eye((self.num_seasons-2)*self.dim_obs),
                            jnp.zeros(((self.num_seasons-2)*self.dim_obs, self.dim_obs))]])}

    @property
    def observation_matrix(self):
        return {self.component_name:
                jnp.block([jnp.eye(self.dim_obs),
                           jnp.zeros((self.dim_obs, (self.num_seasons-2)*self.dim_obs))])}

    @property
    def transition_covariance(self):
        return {'seasonal': self.drift_covariance_prior.mode()}

    @property
    def transition_covariance_prior(self):
        return {'seasonal': self.drift_covariance_prior}

    @property
    def initial_state_prior(self):
        c = self.num_seasons - 1
        initial_loc = jnp.array([self.initial_effect_prior.mean()]*c).flatten()
        initial_cov = jsp.linalg.block_diag(
            *([self.initial_effect_prior.covariance()]*c))
        initial_pri = MVN(loc=initial_loc, covariance_matrix=initial_cov)
        return {'seasonal': initial_pri}

    @property
    def cov_spars_matrix(self):
        return {'seasonal': jnp.concatenate(
                    (
                        jnp.eye(self.dim_obs),
                        jnp.zeros((self.dim_obs*(self.num_seasons-2), self.dim_obs)),
                    ),
                    axis=0)
                }
