from abc import ABC
from abc import abstractmethod

import jax.numpy as jnp
from ssm_jax.distributions import InverseWishart as IW
from ssm_jax.distributions import MatrixNormalPrecision as MN
from ssm_jax.structural_time_series.models.structural_time_series_ssm import StructuralTimeSeriesSSM
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

tfd = tfp.distributions


def _set_prior(input_prior, default_prior):
    return input_prior if input_prior is not None else default_prior


class StructuralTimeSeries():
    """The class of the Bayesian structural time series (STS) model

    The STS model is defined by a pair of equations relating
    a time series observation y_t to
    a vector of latent state z_t:

    y_t     = H_t @ z_t + D_t @ u_t + N(0, R_t)
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
        observed_timeseries: has shape (batch_size, timesteps, dim_observed_timeseries)
        name (str): name of the STS model
    """

    def __init__(self,
                 components,
                 observed_timeseries,
                 observation_covariance=None,
                 observation_covariance_prior=None,
                 name='StructuralTimeSeries'):

        _dim = observed_timeseries.shape
        self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
        obs_scale = jnp.std(observed_timeseries, axis=0).mean()

        self.name = name
        self.component_names = [c.component_name for c in components]

        # Save parameters of the STS model:
        self.initial_state_priors = []

        self.transition_matrices = []
        self.transition_covariances = []
        self.transition_covariance_priors = []

        self.observation_matrices = []
        if observation_covariance is not None:
            self.observation_covariance = observation_covariance
        else:
            self.observation_covariance = jnp.eye(self.dim_obs) * obs_scale**2
        self.observation_regression_weights = None
        self.observation_regression_weights_prior = None
        self.observation_covariance_prior = _set_prior(
            observation_covariance_prior, IW(df=self.dim_obs, scale=jnp.eye(self.dim_obs)))

        # Aggregate components
        for component in components:
            if isinstance(component, LinearRegression):
                self.observation_design_matrix = component.design_matrix
                self.observation_regression_weights_prior = component.weights_prior
            elif isinstance(component, STSLatentComponent):
                self.transition_matrices.append(component.transition_matrix)
                self.observation_matrices.append(component.observation_matrix)

                self.initial_state_priors.extend(component.initial_state_prior)
                self.transition_covariance_priors.extend(component.transition_covariance_prior)

    def as_ssm(self):
        """Formulate the STS model as a linear Gaussian state space model:

        p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
        p(y_t | z_t) = N(y_t | H_t z_t, R_t)
        p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})

        F_t, H_t are fixed known matrices,
        the convariance matrices, Q and R, are random variables to be learned,
        the regression coefficient matrix B is also unknown random matrix
        if the STS model includes an regression component
        """
        sts_ssm = StructuralTimeSeriesSSM(self.transition_matrices, self.observation_matrices,
                                          self.initial_state_priors,
                                          self.transition_covariance_priors,
                                          self.observation_covariance,
                                          self.observation_covariance_prior,
                                          self.observation_regression_weights_prior)
        return sts_ssm

    def fit_hmc(self, key, sample_size, observed_time_series, inputs=None):
        """Sampling parameters of the STS model from their posterior distributions.

        Parameters of the STS model includes:
            covariance matrix of each component,
            covariance matrix of observation,
            regression coefficient matrix (if the model has inputs and a regression component)
        """
        sts_ssm = self.as_ssm()
        samps_lgssm_params = sts_ssm.fit_hmc(key, sample_size, observed_time_series, inputs)
        return samps_lgssm_params

    def forecast(self,
                 observed_time_series,
                 parameter_samples,
                 num_steps_forecast,
                 include_observation_noise=True):
        forecast_ssm = self.as_ssm()
        
    

    def sample(self, key, num_timesteps, inputs=None):
        """Given parameters, sample latent states and corresponding observed time series.
        """
        sts_ssm = self.as_ssm()
        states, timeseries = sts_ssm.sample(key, num_timesteps, inputs)
        component_states = self._split_joint_states(states)
        return component_states, timeseries

    def marginal_log_prob(self, emissions, inputs=None):
        sts_ssm = self.as_ssm()
        return sts_ssm.marginal_log_prob(emissions, inputs)

    def filter(self, emissions, inputs=None):
        sts_ssm = self.as_ssm()
        states = sts_ssm.filter(emissions, inputs)
        component_states = self._split_joint_states(states)
        return component_states

    def smoother(self, emissions, inputs=None):
        sts_ssm = self.as_ssm()
        states = sts_ssm.smoother(emissions, inputs)
        component_states = self._split_joint_states(states)
        return component_states


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
    def transition_covariance_prior(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state_prior(self):
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
                 observed_timeseries=None,
                 dim_observed_timeseries=1,
                 name='LocalLinearTrend'):
        if observed_timeseries is not None:
            _dim = observed_timeseries.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
            obs_scale = jnp.std(observed_timeseries, axis=0).mean()
            obs_init = observed_timeseries[0].mean()
        else:
            self.dim_obs = dim_observed_timeseries
            obs_scale = 1.
            obs_init = 0.

        self.component_name = name

        # Initialize the prior using the observed time series if a prior is not specified
        self.level_covariance_prior = _set_prior(
            level_covariance_prior, IW(df=self.dim_obs, scale=obs_scale * jnp.eye(self.dim_obs)))

        self.slope_covariance_prior = _set_prior(
            slope_covariance_prior, IW(df=self.dim_obs, scale=5. * jnp.eye(self.dim_obs)))

        self.initial_level_prior = _set_prior(
            initial_level_prior,
            MVN(loc=obs_init * jnp.ones(self.dim_obs),
                covariance_matrix=obs_scale * jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_level_prior, MVN)

        self.initial_slope_prior = _set_prior(
            initial_slope_prior,
            MVN(loc=jnp.zeros(self.dim_obs), covariance_matrix=jnp.eye(self.dim_obs)))
        assert isinstance(self.initial_slope_prior, MVN)

    @property
    def transition_matrix(self):
        return jnp.block([[jnp.eye(self.dim_obs), jnp.eye(self.dim_obs)],
                          [jnp.zeros((self.dim_obs, self.dim_obs)),
                           jnp.eye(self.dim_obs)]])

    @property
    def observation_matrix(self):
        return jnp.block([jnp.eye(self.dim_obs), jnp.zeros((self.dim_obs, self.dim_obs))])

    @property
    def transition_covariance_prior(self):
        return [self.level_covariance_prior, self.slope_covariance_prior]

    @property
    def initial_state_prior(self):
        return [self.initial_level_prior, self.initial_slope_prior]


class LinearRegression():
    """The static regression component of the structual time series (STS) model

    if add_bias:
        reg_t = weights @ input_t + bias
    else:
        reg_t = weights @ input_t

    The matrix 'weights' has a MatrixNormal prior

    Args:
        weights_prior: MatrixNormal prior for the weight matrix
        dim_input: Number of explanatory variables (excluding the bias term)
        add_bias (bool): Whether or not to include a bias term in the linear regression model
        dim_observed_time_series: Dimension of the observed time series
        name (str): Name of the component in the STS model
    """

    def __init__(self,
                 design_matrix,
                 weights_prior=None,
                 dim_observed_timeseries=1,
                 name='LinearRegression'):
        self.dim_obs = dim_observed_timeseries
        self.dim_input = design_matrix.shape[-1]
        self.component_name = name

        # Initialize the prior distribution for weights
        if weights_prior is None:
            weights_prior = MN(loc=jnp.zeros((self.dim_obs, self.dim_input)),
                               row_covariance=jnp.eye(self.dim_obs),
                               col_precision=jnp.eye(self.dim_input))

        self.design_matrix = design_matrix
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
                 observed_timeseries=None,
                 dim_observed_timeseries=1,
                 name='Seasonal'):
        if observed_timeseries is not None:
            _dim = observed_timeseries.shape
            self.dim_obs = 1 if len(_dim) == 1 else _dim[-1]
        else:
            self.dim_obs = dim_observed_timeseries

        self.num_seasons = num_seasons
        self.num_steps_per_season = num_steps_per_season
        self.component_name = name

        self.initial_effect_prior = _set_prior(
            initial_effect_prior,
            MVN(loc=jnp.zeros(self.dim_obs), covariance_matrix=jnp.eye(self.dim_obs)))

        self.drift_covariance_prior = _set_prior(
            drift_covariance_prior, IW(df=self.dim_obs + 0.1, scale=jnp.eye(self.dim_obs)))

    @property
    def transition_matrix(self):
        # TODO: allow num_steps_per_season > 1 or be a list of integers
        return jnp.block([[jnp.kron(-jnp.ones(self.num_seasons - 1), jnp.eye(self.dim_obs))],
                          [
                              jnp.eye((self.num_seasons - 2) * self.dim_obs),
                              jnp.zeros(((self.num_seasons - 2) * self.dim_obs, self.dim_obs))
                          ]])

    @property
    def observation_matrix(self):
        return jnp.block([
            jnp.eye(self.dim_obs),
            jnp.zeros((self.dim_obs, (self.num_seasons - 2) * self.dim_obs))
        ])

    @property
    def transition_covariance_prior(self):
        return [self.drift_covariance_prior] + [None] * (self.num_seasons - 2)

    @property
    def initial_state_prior(self):
        return [self.initial_effect_prior] * (self.num_seasons - 1)
