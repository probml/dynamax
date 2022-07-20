import jax.numpy as jnp
import jax.random as jr
from jax import lax
import chex
import tensorflow_probability.substrates.jax.distributions as tfd
MN = tfd.MatrixNormalLinearOperator

from distrax import MultivariateNormalFullCovariance as MVN
from inference import LGSSMParams, lgssm_posterior_sample

from NIW import NormalInverseWishart as NIW
from NIW import InverseWishart as IW


def lgssm_blocked_gibbs(rng, num_itrs, emissions, prior_hyperparams=None, inputs=None, dimension_hidden=None):
    """Estimation using blocked-Gibbs sampler
    
    Assume that parameters are fixed over time

    Args:
        rng (_type_): _description_
        emissions (_type_): _description_
        prior (_type_): _description_
        inputs (_type_, optional): _description_. Defaults to None.
    """
    initial_prior_params, dynamics_prior_params, emission_prior_params = prior_hyperparams
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    D_hid = dimension_hidden
    D_obs = emissions.shape[1]
    D_in = inputs.shapes[1]
    
    def log_prior_prob(params):
        # log prior probability of the initial state
        initial_pri = NIW()
        lp_Sm = initial_pri.log_prob({'mu':params.initial_mean, 'Sigma':params.initial_covariance})
        
        # log prior probability of the dynamics
        Q_pri = IW(dynamics_prior_params[2:])
        FBb_pri = MN(dynamics_prior_params[0],
                     jnp.linalg.cholesky(params.dynamics_covariance),
                     jnp.linalg.inv(jnp.linalg.cholesky(dynamics_prior_params[1])))
        lp_FBb = FBb_pri.log_prob(jnp.hstack((
            params.dynamics_matrix, jnp.vstack((params.dynamics_bias, params.dynamics_input_weights)))))
        lp_Q = Q_pri.log_prob(params.dynamics_covariance)
        
        # log prior probability of the emission
        R_pri = IW(emission_prior_params[2:])
        HDd_pri = MN(emission_prior_params[0],
                     jnp.linalg.cholesky(params.emission_covariance),
                     jnp.linalg.inv(jnp.linalg.cholesky(emission_prior_params[1])))
        lp_HDd = HDd_pri.log_prob(jnp.hstack((params.emission_matrix,
            jnp.vstack((params.emission_bias, params.emission_input_weights)))))
        lp_R = R_pri.log_prob(params.emission_covariance)
        
        return lp_Sm + lp_FBb + lp_Q + lp_HDd + lp_R
    
    def mniw_distribution_update(M_pri, V_pri, nu_pri, Psi_pri, SxxT, SxyT, SyyT, L):
        """Update the MatrixNormalInverseWishart distribution for the dynamics or the emission

        Args:
            M_pri (_type_): _description_
            V_pri (_type_): _description_
            nu_pri (_type_): _description_
            Psi_pri (_type_): _description_

        Returns:
            _type_: _description_
        """
        Sxx = V_pri + SxxT
        Sxy = SxyT + V_pri @ M_pri.T
        Syy = SyyT + M_pri @ V_pri @ M_pri.T
        M_pos = jnp.linalg.solve(Sxx, Sxy).T
        V_pos = Sxx
        nu_pos = nu_pri + L
        Psi_pos = Psi_pri + Syy - M_pos @ Sxy
        return M_pos, V_pos, nu_pos, Psi_pos
    
    def lgssm_params_sample(rng, states, inputs, emissions):
        rngs = iter(jr.split(rng, 8))
        
        # shorthand
        x, xp, xn = states, states[:-1], states[1:]
        u, up= inputs, inputs[:-1]
        y = emissions
        # quantities for the dynamics distribution
        # let zp[t] = [x[t], u[t], 1] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[xp.T @ xp,          xp.T @ up,          xp.sum(0)[:, None]],
                               [up.T @ xp,          up.T @ up,          up.sum(0)[:, None]],
                               [xp.sum(0)[None, :], up.sum(0)[None, :],   num_timesteps-1 ]])
        sum_zpxnT = jnp.block([[xp.T @ xn],
                               [up.T @ xn],
                               [xn.sum(0)[None, :]]])
        sum_xnxnT = xn.T @ xn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps-1)
        # quantities for the emissions
        # let z[t] = [x[t], u[t], 1] for t = 0...T-1
        sum_zzT = jnp.block([[x.T @ x,           x.T @ u,          x.sum(0)[:, None]],
                             [u.T @ x,           u.T @ u,          u.sum(0)[:, None]],
                             [x.sum(0)[None, :], u.sum(0)[None, :],    num_timesteps]])
        sum_zyT = jnp.block([[x.T @ y],
                             [u.T @ y],
                             [y.sum(0)[None,:]]])
        sum_yyT = y.T @ y
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        
        # Sample the initial params
        loc_pri, precision_pri, df_pri, scale_pri = initial_prior_params
        Sm_posterior = NIW((precision_pri*loc_pri+x[0]) / (precision_pri+1),
                           precision_pri + 1,
                           df_pri + 1,
                           scale_pri + jnp.outer(x[0]-loc_pri, x[0]-loc_pri)*precision_pri/(precision_pri+1))
        Sm = Sm_posterior.sample(seed=next(rngs))
        S, m = Sm['Sigma'], Sm['mu']
        
        # Sample the dynamics params
        M_dyn, V_dyn, nu_dyn, Psi_dyn = mniw_distribution_update(*dynamics_prior_params, *dynamics_stats)
        Q = IW(nu_dyn, Psi_dyn).sample(seed=next(rngs))
        FBb = MN(M_dyn, 
                 jnp.linalg.cholesky(Q), 
                 jnp.linalg.inv(jnp.linalg.cholesky(V_dyn))).sample(seed=next(rngs))
        F, B, b = FBb[:, :D_hid], FBb[:, D_hid:-1], FBb[:, -1]
        
        # Sample the emission params
        M_ems, V_ems, nu_ems, Psi_ems = mniw_distribution_update(*emission_prior_params, *emission_stats)
        R = IW(nu_ems, Psi_ems).sample(seed=next(rngs))
        HDd = MN(M_ems, 
                 jnp.linalg.cholesky(R), 
                 jnp.linalg.inv(jnp.linalg.cholesky(V_ems))).sample(seed=next(rngs))
        H, D, d = HDd[:, :D_hid], HDd[:, D_hid:-1], HDd[:, -1]
        
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
    
    def one_sample(params_current, rng):
        rngs = jr.split(rng, 2)
        ll, states = lgssm_posterior_sample(rngs[0], params_current, emissions, inputs)
        params_new = lgssm_params_sample(rngs[1], params_current, states, inputs, emissions)
        l_prior = log_prior_prob(params_new)
        log_evidence = ll + l_prior
        return (params_new, log_evidence), params_new
    
    # Initialize the parameters from the prior
    if prior_hyperparams is None:
        initial_prior_params = jnp.zeros(D_hid), 1., D_hid-1., 1e4*jnp.eye(D_hid)
        dynamics_prior_params = (jnp.zeros((D_hid, D_hid+D_in)), 
                                 jnp.eye(D_hid+D_in), 
                                 D_hid-1., 
                                 1e4 * jnp.eye(D_hid))
        emission_prior_params = (jnp.zeros((D_obs, D_hid+D_in)), 
                                 jnp.eye(D_hid+D_in), 
                                 D_obs-1., 
                                 1e4 * jnp.eye(D_obs))
    
    rng, *rngs = jr.split(rng, 5+1)
    rngs = iter(rngs)
    # Initialize the initial state
    Sm0 = NIW(*initial_prior_params).sample(seed=next(rngs))
    S_0, m_0 = Sm0['Sigma'], Sm0['mu']
    # Initialize the dynamics parameters
    Q_0 = IW(dynamics_prior_params[2:]).sample(seed=next(rngs))
    FBb_0 = MN(dynamics_prior_params[0], 
               jnp.linalg.cholesky(Q_0), 
               jnp.linalg.inv(jnp.linalg.cholesky(dynamics_prior_params[1]))).sample(seed=next(rngs))
    F_0, B_0, b_0 = FBb_0[:, :D_hid], FBb_0[:, D_hid:-1], FBb_0[:, -1]
    # Initialize the emission parameters
    R_0 = IW(emission_prior_params[2:]).sample(seed=next(rngs))
    HDd_0 = MN(emission_prior_params[0], 
               jnp.linalg.cholesky(R_0), 
               jnp.linalg.inv(jnp.linalg.cholesky(emission_prior_params[1]))).sample(seed=next(rngs)) 
    H_0, D_0, d_0 = HDd_0[:, :D_hid], HDd_0[:, D_hid:-1], HDd_0[:, -1]
    
    params_0 = LGSSMParams(initial_mean = m_0,
                           initial_covariance = S_0,
                           dynamics_matrix = F_0,
                           dynamics_input_weights = B_0,
                           dynamics_bias = b_0,
                           dynamics_covariance = Q_0,
                           emission_matrix = H_0,
                           emission_input_weights = D_0,
                           emission_bias = d_0,
                           emission_covariance = R_0)
    
    # Sample
    rngs = jr.split(rng, num_itrs)
    params_samples, log_probs = lax.scan(one_sample, params_0, rngs)
    
    return params_samples, log_probs