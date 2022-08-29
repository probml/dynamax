from functools import partial
from tqdm.auto import trange

from jax import numpy as jnp
from jax import random as jr
from jax import lax, vmap, jit
from jax.tree_util import tree_map, register_pytree_node_class

import blackjax
from distrax import MultivariateNormalFullCovariance as MVN

from ssm_jax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother
from ssm_jax.distributions import NormalInverseWishart as NIW, \
    MatrixNormalInverseWishart as MNIW
from ssm_jax.distributions import niw_posterior_update, mniw_posterior_update
from ssm_jax.utils import PSDToRealBijector
from ssm_jax.abstractions import SSM, Parameter



_get_shape = lambda x, dim: x.shape[1:] if x.ndim == dim + 1 else x.shape

@register_pytree_node_class
class LinearGaussianSSM(SSM):
    """
    Linear Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | F_t z_{t-1} + B_t u_t + b_t, Q_t)
    p(y_t | z_t) = N(y_t | H_t z_t + D_t u_t + d_t, R_t)
    p(z_1) = N(z_1 | mu_{1|0}, Sigma_{1|0})
    where z_t = hidden, y_t = observed, u_t = inputs,
    dynamics_matrix = F
    dynamics_covariance = Q
    emission_matrix = H
    emissions_covariance = R
    initial_mean = mu_{1|0}
    initial_covariance = Sigma_{1|0}
    Optional parameters (default to 0)
    dynamics_input_matrix = B
    dynamics_bias = b
    emission_input_matrix = D
    emission_bias = d
    """

    def __init__(
        self,
        dynamics_matrix,
        dynamics_covariance,
        emission_matrix,
        emission_covariance,
        initial_mean=None,
        initial_covariance=None,
        dynamics_input_weights=None,
        dynamics_bias=None,
        emission_input_weights=None,
        emission_bias=None,
        priors = None
    ):
        self.emission_dim, self.state_dim = _get_shape(emission_matrix, 2)
        dynamics_input_dim = dynamics_input_weights.shape[1] if dynamics_input_weights is not None else 0
        emission_input_dim = emission_input_weights.shape[1] if emission_input_weights is not None else 0
        self.input_dim = max(dynamics_input_dim, emission_input_dim)
        self._db_indicator = dynamics_bias is not None
        self._eb_indicator = emission_bias is not None
        
        # Set optional args to default value if not given
        default = lambda x, v: x if x is not None else v
        
        initial_mean = default(initial_mean, jnp.zeros(self.state_dim))
        initial_covariance = default(initial_covariance, jnp.eye(self.state_dim))
        dynamics_input_weights = default(dynamics_input_weights, jnp.zeros((self.state_dim, self.input_dim)))
        dynamics_bias = default(dynamics_bias, jnp.zeros(self.state_dim))
        emission_input_weights = default(emission_input_weights, jnp.zeros((self.emission_dim, self.input_dim)))
        emission_bias = default(emission_bias, jnp.zeros(self.emission_dim))
        
        # Save args
        self._dynamics_matrix = Parameter(dynamics_matrix)
        self._dynamics_covariance = Parameter(dynamics_covariance, bijector=PSDToRealBijector)
        self._emission_matrix = Parameter(emission_matrix)
        self._emission_covariance = Parameter(emission_covariance, bijector=PSDToRealBijector) 
        self._initial_mean = Parameter(initial_mean)
        self._initial_covariance = Parameter(initial_covariance, bijector=PSDToRealBijector)
        self._dynamics_input_weights = Parameter(dynamics_input_weights)
        self._dynamics_bias = Parameter(dynamics_bias)
        self._emission_input_weights = Parameter(emission_input_weights)
        self._emission_bias = Parameter(emission_bias) 

        # Initialize prior distributions 
        if priors is None:
            self.initial_prior = self.dynamics_prior = self.emission_prior = None
        else: 
            self.initial_prior, self.dynamics_prior, self.emission_prior = priors
            
        if self.initial_prior is None:
            self.initial_prior = NIW(loc=self.initial_mean,
                                     mean_concentration=1.,
                                     df=self.state_dim,
                                     scale=jnp.eye(self.state_dim))
            
        if self.dynamics_prior is None:
            loc_d = self._join_matrix(self.dynamics_matrix, 
                                      self.dynamics_input_weights, 
                                      self.dynamics_bias,
                                      self._db_indicator)
            self.dynamics_prior = MNIW(loc=loc_d, 
                                       col_precision=jnp.eye(loc_d.shape[1]),
                                       df=self.state_dim, 
                                       scale=jnp.eye(self.state_dim))
            
        if self.emission_prior is None:
            loc_e = self._join_matrix(self.emission_matrix,
                                      self.emission_input_weights,
                                      self.emission_bias,
                                      self._eb_indicator)
            self.emission_prior = MNIW(loc=loc_e, 
                                       col_precision=jnp.eye(loc_e.shape[1]),
                                       df=self.emission_dim, 
                                       scale=jnp.eye(self.emission_dim))

        # Check shapes
        assert self.initial_mean.shape == (self.state_dim,)
        assert self.initial_covariance.shape == (self.state_dim, self.state_dim)
        assert self.dynamics_matrix.shape[-2:] == (self.state_dim, self.state_dim)
        assert self.dynamics_input_weights.shape[-2:] == (self.state_dim, self.input_dim)
        assert self.dynamics_bias.shape[-1:] == (self.state_dim,)
        assert self.dynamics_covariance.shape == (self.state_dim, self.state_dim)
        assert self.emission_input_weights.shape[-2:] == (self.emission_dim, self.input_dim)
        assert self.emission_bias.shape[-1:] == (self.emission_dim,)
        assert self.emission_covariance.shape == (self.emission_dim, self.emission_dim)
        
        self.param_keys = ["_initial_mean", "_initial_covariance",
                           "_dynamics_matrix", "_dynamics_input_weights", "_dynamics_bias", "_dynamics_covariance",
                           "_emission_matrix", "_emission_input_weights", "_emission_bias", "_emission_covariance"]

    @classmethod
    def random_initialization(cls, key, state_dim, emission_dim, input_dim=0):
        m1 = jnp.zeros(state_dim)
        Q1 = jnp.eye(state_dim)
        # TODO: Sample a random rotation matrix
        F = 0.99 * jnp.eye(state_dim)
        B = jnp.zeros((state_dim, input_dim))
        Q = 0.1 * jnp.eye(state_dim)
        H = jr.normal(key, (emission_dim, state_dim))
        D = jnp.zeros((emission_dim, input_dim))
        R = 0.1 * jnp.eye(emission_dim)
        return cls(
            dynamics_matrix=F,
            dynamics_covariance=Q,
            emission_matrix=H,
            emission_covariance=R,
            initial_mean=m1,
            initial_covariance=Q1,
            dynamics_input_weights=B,
            emission_input_weights=D
        )
        
    # Properties to get various parameters of the model
    # Parameters of initial state
    @property
    def initial_mean(self):
        return self._initial_mean.value
    
    @property 
    def initial_covariance(self):
        return self._initial_covariance.value
    
    # Parameters of dynamics model
    @property
    def dynamics_matrix(self):
        return self._dynamics_matrix.value
    
    @property
    def dynamics_input_weights(self):
        return self._dynamics_input_weights.value
    
    @property
    def dynamics_bias(self):
        return self._dynamics_bias.value
    
    @property 
    def dynamics_covariance(self):
        return self._dynamics_covariance.value
    
    # Parameters of emission model
    @property
    def emission_matrix(self):
        return self._emission_matrix.value
    
    @property 
    def emission_input_weights(self):
        return self._emission_input_weights.value
    
    @property
    def emission_bias(self):
        return self._emission_bias.value
    
    @property
    def emission_covariance(self):
        return self._emission_covariance.value
    
    def initial_distribution(self):
        return MVN(self.initial_mean, self.initial_covariance)
    
    def transition_distribution(self, state, **covariates):
        if input not in covariates:
            input = jnp.zeros(self.input_dim)
        return MVN(self.dynamics_matrix @ state + self.dynamics_input_weights @ input + self.dynamics_bias,
                   self.dynamics_covariance)
    
    def emission_distribution(self, state, **covariates):
        if input not in covariates:
            input = jnp.zeros(self.input_dim)
        return MVN(self.emission_matrix @ state + self.emission_input_weights @ input + self.emission_bias,
                   self.emission_covariance)

    def log_prior(self):
        """Return the log prior probability of any model parameters.
        Returns:
            lp (Scalar): log prior probability.
        """
        d_matrix = self._join_matrix(self.dynamics_matrix, 
                                     self.dynamics_input_weights, 
                                     self.dynamics_bias,
                                     self._db_indicator)
        e_matrix = self._join_matrix(self.emission_matrix, 
                                     self.emission_input_weights, 
                                     self.emission_bias,
                                     self._eb_indicator)
        # Compute log probs
        lp = self.initial_prior.log_prob((self.initial_covariance, self.initial_mean))
        lp += self.dynamics_prior.log_prob((self.dynamics_covariance, d_matrix))
        lp += self.emission_prior.log_prob((self.emission_covariance, e_matrix))
        return lp

    def marginal_log_prob(self, emissions, inputs=None):
        """Compute log marginal likelihood of observations."""
        filtered_posterior = lgssm_filter(self, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(self, emissions, inputs=None):
        """Compute filtering distribution."""
        return lgssm_filter(self, emissions, inputs)

    def smoother(self, emissions, inputs=None):
        """Compute smoothing distribution."""
        return lgssm_smoother(self, emissions, inputs)
    
    # TODO: currently the set params does not support frozen
    @property
    def params(self):
        # Find all parameters
        params = [self.__dict__[key].value for key in self.param_keys]
        return params

    @params.setter
    def params(self, values):
        assert len(self.param_keys) == len(values)
        for key, value in zip(self.param_keys, values):
            self.__dict__[key].value = value
    
    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_mean = unconstrained_params[0]
        initial_covariance = PSDToRealBijector.inverse(unconstrained_params[1])
        dynamics_matrix = unconstrained_params[2]
        dynamics_input_weights = unconstrained_params[3]
        dynamics_bias = unconstrained_params[4]
        dynamics_covariance = PSDToRealBijector.inverse(unconstrained_params[5])
        emission_matrix = unconstrained_params[6]
        emission_input_weights = unconstrained_params[7]
        emission_bias = unconstrained_params[8]
        emission_covariance = PSDToRealBijector.inverse(unconstrained_params[9])
        return cls(
            dynamics_matrix=dynamics_matrix,
            dynamics_covariance=dynamics_covariance,
            emission_matrix=emission_matrix,
            emission_covariance=emission_covariance,
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_input_weights=dynamics_input_weights,
            dynamics_bias=dynamics_bias,
            emission_input_weights=emission_input_weights,
            emission_bias=emission_bias
        )
        
    ### Expectation-maximization (EM) code
    def e_step(self, batch_emissions, batch_inputs=None):
        """The E-step computes sums of expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """
        num_batches, num_timesteps = batch_emissions.shape[:2]
        if batch_inputs is None:
            batch_inputs = jnp.zeros((num_batches, num_timesteps, 0))

        def _single_e_step(emissions, inputs):
            # Run the smoother to get posterior expectations
            posterior = lgssm_smoother(self, emissions, inputs)

            # shorthand
            Ex = posterior.smoothed_means
            Exp = posterior.smoothed_means[:-1]
            Exn = posterior.smoothed_means[1:]
            Vx = posterior.smoothed_covariances
            Vxp = posterior.smoothed_covariances[:-1]
            Vxn = posterior.smoothed_covariances[1:]
            Expxn = posterior.smoothed_cross_covariances
            # Append bias to the inputs
            inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            up = inputs[:-1]
            u = inputs
            y = emissions

            # expected sufficient statistics for the initial distribution
            Ex0 = posterior.smoothed_means[0]
            Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
            init_stats = (Ex0, Ex0x0T, 1)

            # expected sufficient statistics for the dynamics distribution
            # let zp[t] = [x[t], u[t]] for t = 0...T-2
            # let xn[t] = x[t+1]          for t = 0...T-2
            sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up],
                                   [ up.T @ Exp,  up.T @ up]])
            sum_zpzpT = sum_zpzpT.at[: self.state_dim, : self.state_dim].add(Vxp.sum(0))
            sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
            sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
            if not self._db_indicator:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1,:], sum_xnxnT, num_timesteps - 1)

            # more expected sufficient statistics for the emissions
            # let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[Ex.T @ Ex,  Ex.T @ u],
                                 [ u.T @ Ex,   u.T @ u]])
            sum_zzT = sum_zzT.at[: self.state_dim, : self.state_dim].add(Vx.sum(0))
            sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
            sum_yyT = emissions.T @ emissions
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
            if not self._eb_indicator:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1,:], sum_yyT, num_timesteps)

            return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

        # TODO: what's the best way to vectorize/parallelize this?
        return vmap(_single_e_step)(batch_emissions, batch_inputs)
    
    def m_step(self, batch_stats):
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = jnp.linalg.solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # initial distribution
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        # dynamics distribution
        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self._db_indicator \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        # emission distribution
        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self._eb_indicator \
            else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))
        
        self._initial_mean.value = m
        self._initial_covariance.value = S
        self._dynamics_matrix.value = F
        self._dynamics_input_weights.value = B
        self._dynamics_bias.value = b
        self._dynamics_covariance.value = Q
        self._emission_matrix.value = H
        self._emission_input_weights.value = D
        self._emission_bias.value = d
        self._emission_covariance.value = R

    def fit_em(self, batch_emissions, batch_inputs=None, num_iters=50, method='MLE'):
        """Fit this HMM with Expectation-Maximization (EM).
        """
        @jit
        def em_step(_params):
            self.params = _params
            posterior_stats, marginal_loglikes = self.e_step(batch_emissions, batch_inputs)
            self.m_step(posterior_stats)     
            _params = self.params
            return _params, marginal_loglikes.sum()

        log_probs = []
        _params = self.params
        
        for _ in trange(num_iters):
            _params, marginal_loglik = em_step(_params) 
            log_probs.append(marginal_loglik)
        
        self.params = _params
        return jnp.array(log_probs)
    
    def _join_matrix(self, F, B, b, indicator):
        """Join the transition matrix and weight matrix so that they can be inferred jointly.
        
        If indicator=True, 
        the bias vector b is added to the last column of the [F, B] matrix,
        otherwise,
        there is NO bias term, and F, B are joint.
        """
        if not indicator:
            return jnp.concatenate((F, B), axis=1)
        else:
            return jnp.concatenate((F, B, b[:,None]), axis=1)