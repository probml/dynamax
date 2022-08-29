import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap

from ssm_jax.linear_gaussian_ssm.inference import LGSSMParams, lgssm_posterior_sample
from ssm_jax.distributions import NormalInverseWishart as NIW, MatrixNormalInverseWishart as MNIW
from ssm_jax.distributions import niw_posterior_update, mniw_posterior_update


def blocked_gibbs(key, sample_size, emissions, D_hid, 
                  priors=None, 
                  inputs=None,
                  dynamics_bias_indicator=False,
                  emission_bias_indicator=False):
    """Estimation using blocked-Gibbs sampler
    
    Assume that parameters are fixed over time

    Args:
        key:          an object of jax.random.PRNGKey
        sample_size:  number of samples from the Gibbs sampler
        emissions:    a sequence of observations 
        priors:       a tuple containing prior distributions of the model 
                      priors = (initial_prior, 
                                dynamics_prior, 
                                emission_prior), 
                      where 
                      initial_prior is an NIW object
                      dynamics_prior is an MNIW object
                      emission_prior is an MNIW object
        D_hid:        dimension of hidden state
        inputs:       has shape (num_timesteps, dim_input) or None
        dynamics_bias_indicator: whether to include a bias term in dynamics model
        emission_bias_indicator: whether to include a bias term in emission model
    """
    num_timesteps = len(emissions)
    D_obs = emissions.shape[1]
    if inputs is None:
        inputs = jnp.zeros((num_timesteps, 0))
    D_in = inputs.shape[1]
    
    # Set default priors using observations
    # this is similar to empirical Bayesian approach
    scale_obs = jnp.std(emissions, axis=0).mean()
    if priors is None:
        initial_prior = dynamics_prior = emission_prior = None
    else: 
        initial_prior, dynamics_prior, emission_prior = priors
        
    if initial_prior is None:
        initial_mean = jnp.ones(D_hid) * emissions[0].mean()
        initial_prior = NIW(loc=initial_mean,
                            mean_concentration=1.,
                            df=D_hid,
                            scale=5.*scale_obs*jnp.eye(D_hid))
        
    if dynamics_prior is None:
        F = jnp.ones((D_hid, D_hid)) / D_hid
        B = 0.1 * jr.uniform(key, shape=[D_hid, D_in])
        loc_d = jnp.concatenate((F, B), axis=1)
        if dynamics_bias_indicator:
            loc_d = jnp.concatenate((loc_d, jnp.zeros((D_hid, 1))), axis=1)
        dynamics_prior = MNIW(loc=loc_d, 
                              col_precision=jnp.eye(loc_d.shape[1]),
                              df=D_hid, 
                              scale=5.*jnp.eye(D_hid))
        
    if emission_prior is None:
        H = jnp.ones((D_obs, D_hid)) / D_hid
        D = 0.1 * jr.uniform(key, shape=[D_obs, D_in])
        loc_e = jnp.concatenate((H, D), axis=1)
        if emission_bias_indicator:
            loc_e = jnp.concatenate((loc_e, jnp.zeros((D_obs, 1))), axis=1)
        emission_prior = MNIW(loc=loc_e, 
                              col_precision=jnp.eye(loc_e.shape[1]),
                              df=D_obs, 
                              scale=5.*jnp.eye(D_obs))
    
    # Check dimensions
    assert dynamics_prior.mode()[1].shape == (D_hid, D_hid+D_in+dynamics_bias_indicator)
    assert emission_prior.mode()[1].shape == (D_obs, D_hid+D_in+emission_bias_indicator)
    
    def log_prior_prob(params_initial, params_dynamics, params_emission):
        lp_init =  initial_prior.log_prob(params_initial)
        lp_dyn  = dynamics_prior.log_prob(params_dynamics)
        lp_ems  = emission_prior.log_prob(params_emission)
        return lp_init + lp_dyn + lp_ems 
    
    def sufficient_stats_from_sample(states):
        """Convert samples of states to sufficient statistics
        
        Returns:
            (initial_stats, dynamics_stats, emission_stats)
        """
        inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        # let xn[t] = x[t+1]          for t = 0...T-2
        x, xp, xn = states, states[:-1], states[1:]
        u, up= inputs_joint, inputs_joint[:-1]
        y = emissions

        init_stats = (x[0], jnp.outer(x[0], x[0]), 1)
        
        # quantities for the dynamics distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        sum_zpzpT = jnp.block([[xp.T @ xp,  xp.T @ up],
                               [up.T @ xp,  up.T @ up]])
        sum_zpxnT = jnp.block([[xp.T @ xn],
                               [up.T @ xn]])
        sum_xnxnT = xn.T @ xn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps-1)
        if not dynamics_bias_indicator:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1,:], sum_xnxnT, num_timesteps-1)
        
        # quantities for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[x.T @ x,  x.T @ u],
                             [u.T @ x,  u.T @ u]])
        sum_zyT = jnp.block([[x.T @ y],
                             [u.T @ y]])
        sum_yyT = y.T @ y
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not emission_bias_indicator:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1,:], sum_yyT, num_timesteps)
        
        return init_stats, dynamics_stats, emission_stats
        
    def lgssm_params_sample(rng, init_stats, dynamics_stats, emission_stats):
        """Sample parameters of the model.
        """
        rngs = iter(jr.split(rng, 3))
        
        # Sample the initial params
        initial_posterior = niw_posterior_update(initial_prior, init_stats)
        S, m = initial_posterior.sample(seed=next(rngs))
        
        # Sample the dynamics params
        dynamics_posterior = mniw_posterior_update(dynamics_prior, dynamics_stats)
        Q, FB = dynamics_posterior.sample(seed=next(rngs))
        
        # Sample the emission params
        emission_posterior = mniw_posterior_update(emission_prior, emission_stats)
        R, HD = emission_posterior.sample(seed=next(rngs))
            
        return (S, m), (Q, FB), (R, HD)
    
    def _params_wrapper(params_initial, params_dynamics, params_emission):
        S, m = params_initial
        
        Q, FB = params_dynamics
        F = FB[:, :D_hid]
        B, b = (FB[:, D_hid:-1], FB[:, -1]) if dynamics_bias_indicator \
            else (FB[:, D_hid:], jnp.zeros(D_hid)) 
        
        R, HD = params_emission
        H = HD[:, :D_hid]
        D, d = (HD[:, D_hid:-1], HD[:, -1]) if emission_bias_indicator \
            else (HD[:, D_hid:], jnp.zeros(D_obs))
        
        return LGSSMParams(initial_mean = m,
                           initial_covariance = S,
                           dynamics_matrix = F,
                           dynamics_input_weights = B,
                           dynamics_bias = b,
                           dynamics_covariance = Q,
                           emission_matrix = H,
                           emission_input_weights = D,
                           emission_bias = d,
                           emission_covariance = R)
    
    def one_sample(params, rng):
        """One complete iteration of the blocked Gibbs sampler
        """
        rngs = jr.split(rng, 2)
        l_prior = log_prior_prob(*params)
        wrap_params = _params_wrapper(*params)
        ll, states = lgssm_posterior_sample(rngs[0], wrap_params, emissions, inputs)
        
        # Compute sufficient statistics for parameters
        _stats = sufficient_stats_from_sample(states)
        
        # Sample parameters
        params_new = lgssm_params_sample(rngs[1], *_stats)
        
        # Compute the log probability
        log_probs = l_prior + ll
        
        return params_new, (params, log_probs)
    
    # Initialize parameters
    params_initial  =  initial_prior.mode()
    params_dynamics = dynamics_prior.mode()
    params_emission = emission_prior.mode()
    params_0 = (params_initial, params_dynamics, params_emission)
    
    # Sample
    keys = jr.split(key, sample_size)
    _, samples_and_log_probs = lax.scan(one_sample, params_0, keys)
    sample_of_params, log_probs = samples_and_log_probs
    sample_of_params = vmap(_params_wrapper)(*sample_of_params)
    
    return log_probs, sample_of_params
