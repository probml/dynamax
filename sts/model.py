import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax import lax

from ssm_jax.lgssm.inference import lgssm_posterior_sample
from ssm_jax.lgssm.models import LinearGaussianSSM


class StructualTimeSeries():
    """The class of the Bayesian structual time series (STS) model 
    
    The STS model is defined by a pair of equations relating 
    a time series observation y_t to 
    a vector of latent state z_t:
    
    y_t     = H @ z_t + B @ u_t + \epsilon_{y,t},      \epsilon_{y,t} \sim MultivariateNormal(0, R)
    z_{t+1} = F @ z_t +           \epsilon_{z,t},      \epsilon_{z,t} \sim MultivariateNormal(0, Q)
    
    H: fixed emission matrix
    B: random matrix of regression coefficients
    F: fixed dynamics matrix
    R: random covariance matrix of the observation noise 
    Q: random covariance matrix of the latent state
    """ 
    def __init__(self,
                 components,
                 observation_noise_scale_prior=None,
                 observed_timeseries=None,
                 name='StructuralTimeSeries'):
        """Construct a structural time series (STS) model from a list of components

        Args:
            components:                    list of components
            observation_noise_scale_prior: hyper parameters for the InverseWishart prior
            observed_timeseries:           has shape (batch_size, timesteps, dim_observed_timeseries)
            name (str):                    name of the STS model
        """
        self.name = name
        self.component_Sigma_priors = []
        self.component_initial_priors = []
        self.component_names = []
        _dyn_matrices = []
        _ems_matrices = []
        
        # aggregate components
        for component in components:
            self.component_names.append(component.component_name)
            if isinstance(component, LinearRegression):
                self.reg_coeffs = jnp.zeros((component.dim, component.dim_input))
                self.reg_spike_prior = component.spike_prior
                self.reg_slab_prior = component.slab_prior
            else:
                _dyn_matrices.append(component.dynamics_matrix)
                _ems_matrices.append(component.emission_matrix)
                self.component_Sigma_priors.extend(component.Sigma_prior)
                self.component_initial_priors.extend(component.initial_prior)
                
        if len(_dyn_matrices) > 0:
            self.dynamics_matrix = jsp.linalg.block_diag(*_dyn_matrices)
            self.emission_matrix = jnp.array(_ems_matrices)
    
    def _make_state_space_model(self):
        ssm = LinearGaussianSSM(self.dynamics_matrix, dynamics_covariance,
                                self.emission_matrix, emission_covariance,
                                initial_mean, initial_covariance,
                                dynamics_input_weights,
                                dynamics_bias,
                                emission_input_weights,
                                emission_bias)
        return ssm
    
    def joint_log_prob(self, observed_time_series):
        log_prior = None
        log_lik = None
        return log_prior + log_lik

    def posterior_inference(self, seed, num_samples, observed_time_series, inputs):
        
        ssm_model = self._make_state_space_model()
        
        def single_sample(parameters, key):
            Sigma_latent, Sigma_obs, spikes, reg_coefficients = parameters
            # sample latent state
            _, state = lgssm_posterior_sample(key, ssm_model, observed_time_series, inputs)

            # sample parameters (Sigma) of the each latent component
            
            # sample spike parameter

            # sample regression coefficient

            # sample observation noise level (Sigma_obs)
              
            return (Sigma_latent, Sigma_obs, spikes, reg_coefficients),\
                   (Sigma_latent, Sigma_obs, spikes, reg_coefficients)

        # initialize parameters using prior
        Sigma_latent = None
        Sigma_obs = None
        spikes = None
        reg_coefficients = None
        params_initialization = (Sigma_latent, Sigma_obs, spikes, reg_coefficients)
        
        keys = jr.split(seed, num_samples)
        _, samples_of_parameters = lax.scan(single_sample, params_initialization, keys)
        
        return samples_of_parameters


class LocalLinearTrend():
    def __init__(self, 
                 level_Sigma_prior=None,
                 slope_Sigma_prior=None,
                 initial_level_prior=None,
                 initial_slope_prior=None,
                 observed_timeseries=None,
                 dim_observed_timeseries=1,
                 name='LocalLinearTrend'):
        """The local linear trend component of the structual time series (STS) model
        
        level[t] = level[t-1] + slope[t-1] + MultivariateNormal(zeros(dim_observed_timeseries),
                                                                level_Sigma)
        slope[t] = slope[t-1] + MultivariateNormal(zeros(dim_observed_timeseries), 
                                                   slope_Sigma)

        Args:
            level_Sigma_prior:        hyper parameters for the InverseWishart prior
            slope_Sigma_prior:        hyper parameters for the InverseWishart prior
            initial_level_prior:      hyper parameters for the MultivariateNormal prior
            initial_slope_prior:      hyper parameters for the MultivariateNormal prior
            observed_time_series:     has shape (batch_size, timesteps, dim_observed_timeseries)
            dim_observed_time_series: dimension of the observed time series
            name (str):               name of the component in the STS model
        """
        if observed_timeseries is not None:
            self.dim = observed_timeseries.shape[-1]
        else:
            self.dim = dim_observed_timeseries
        
        self.component_name = name
        self.dynamics_matrix = jnp.block([[jnp.eye(self.dim),               jnp.eye(self.dim)],
                                          [jnp.zerso((self.dim, self.dim)), jnp.eye(self.dim)]])
        self.emission_matrix = jnp.block([jnp.eye(self.dim), jnp.eye(self.dim)])
        # TODO: initialize the prior using the observed time series if a prior is not specified
        self.Sigma_prior = [level_Sigma_prior, slope_Sigma_prior]
        self.initial_prior = [initial_level_prior, initial_slope_prior]


class Seasonal():
    def __init__(self, 
                 num_seasons,
                 drift_Sigma_prior=None,
                 initial_effect_prior=None,
                 observed_timeseries=None,
                 dim_observed_timeseries=1,
                 name='Seasonal'):
        """The seasonal component of the structual time series (STS) model
        
        effect_t = - \sum_{k=1}^{num_seasons-1} effect_{t-k} 
                   + MultivaraiteNormal(zeros(dim_observed_timeseries), drift_Sigma)

        Args:
            num_seasons (int):        number of seasons (assuming number of steps per season is 1)
            drift_Sigma_prior:        hyper parameters for the InverseWishart prior
            initial_effect_prior:     hyper parameters for the MultivariateNormal prior
            observed_time_series:     has shape (batch_size, timesteps, dim_observed_timeseries)
            dim_observed_time_series: dimension of the observed time series
            name (str):               name of the component in the STS model
        """
        if observed_timeseries is not None:
            self.dim = observed_timeseries.shape[-1]
        else:
            self.dim = dim_observed_timeseries
            
        self.num_seasons = num_seasons
        self.component_name = name
        _d = (num_seasons-1)*self.dim
        self.dynamics_matrix = jnp.block([[jnp.kron(-jnp.ones(num_seasons), jnp.eye(self.dim))],
                                          [jnp.eye(_d), jnp.zeros((_d, 1))]])
        self.emission_matrix = jnp.block([jnp.eye(self.dim), jnp.zeros(self.dim, _d)])
        # TODO: initialize the prior using the observed time series if a prior is not specified
        self.Sigma_prior = [drift_Sigma_prior, jnp.zeros(self.dim, _d)]
        self.initial_prior = [initial_effect_prior, jnp.zeros(_d)]


class LinearRegression():
    def __init__(self, 
                 design_matrix, 
                 weights_spike_prior=None,
                 weights_slab_prior=None,
                 dim_observed_timeseries=1,
                 name='LinearRegression'):
        """The static regression component of the structual time series (STS) model
        
        reg_t = weights @ input_t,
        The matrix 'weights' has a spike-slab prior, whose posterior can select variables
        
        Args:
            design_matrix:            has shape (batch_size, timesteps, dim_input)
            weights_spike_prior:      hyper parameters for the Bernoulli (spike) distributions of 
                                      indicator of each entry of the weight matrix, 
                                      which has shape (dim_observed_timeseries, dim_input)
            weights_slab_prior:       hyper parameters for the weak MatrixNormal (slab) distributions
                                      of the weight matrix 
            dim_observed_time_series: dimension of the observed time series
            name (str):               name of the component in the STS model
        """
        self.dim = dim_observed_timeseries
        self.dim_input = design_matrix.shape[-1]
        self.component_names = name
        self.design_matrix = design_matrix
        
        # TODO: specify the weights spike-slab prior
        if weights_spike_prior is None:
            # use all inputs in the regression model if no spike prior is specified
            weights_spike_prior = jnp.ones((self.dim, self.dim_input))
            
        self.spike_prior = weights_spike_prior
        self.slab_prior = weights_slab_prior
        