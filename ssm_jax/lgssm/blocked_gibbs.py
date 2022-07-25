import jax.numpy as jnp
import jax.random as jr
from jax import lax

from ssm_jax.lgssm.inference import LGSSMParams, lgssm_posterior_sample

from ssm_jax.utils_distributions import NormalInverseWishart as NIW, MatrixNormalInverseWishart as MNIW


def blocked_gibbs(rng, num_itrs, emissions, prior_hyperparams=None, inputs=None, D_hid=None):
    """Estimation using blocked-Gibbs sampler
    
    Assume that parameters are fixed over time

    Args:
        rng:               an object of jax.random.PRNGKey
        num_itrs:          number of samples from the Gibbs sampler
        emissions:         a sequence of observations 
        prior_hyperparams: a tuple containing hyperparameters of the prior distribution of the model 
                           prior_hyperparams = (initial_prior_params, 
                                                dynamics_prior_params, 
                                                emission_prior_params), 
                           where 
                           initial_prior_params = (loc, mean_precision, df, scale) 
                                is a tuple of parameters of an NIW object
                           dynamics_prior_params = (loc, col_precision_matrix, df, scale) 
                                is a tuple of parameters of an MNIW object
                           emission_prior_params = (loc, col_precision_matrix, df, scale)
                                is a tuple of parameters of an MNIW object
        D_hid:             dimension of hidden state. Not needed if prior_hyperparams is not None,
                           otherwise D_hid must be provided
        inputs:            inputs
    """
    assert prior_hyperparams is not None or D_hid is not None, \
           "prior_hyperparams and D_hid should not both be None"
    
    if prior_hyperparams is not None:
        initial_prior_params, dynamics_prior_params, emission_prior_params = prior_hyperparams
        D_hid = initial_prior_params[0].shape[0]
    else:
        # Set hyperparameters for the prior
        initial_prior_params = jnp.zeros(D_hid), 1., D_hid, 1e4*jnp.eye(D_hid)
        dynamics_prior_params = (jnp.zeros((D_hid, D_hid+D_in+1)), 
                                 jnp.eye(D_hid+D_in+1), 
                                 D_hid, 
                                 1e4 * jnp.eye(D_hid))
        emission_prior_params = (jnp.zeros((D_obs, D_hid+D_in+1)), 
                                 jnp.eye(D_hid+D_in+1), 
                                 D_obs, 
                                 1e4 * jnp.eye(D_obs))
        
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    D_obs = emissions.shape[1]
    D_in = inputs.shape[1]
    
    def log_prior_prob(params):
        """log probability of the model parameters under the prior distributions

        Args:
            params: model parameters

        Returns:
            log probability
        """
        # log prior probability of the initial state
        initial_pri = NIW(*initial_prior_params)
        lp_init = initial_pri.log_prob({'mu':params.initial_mean, 
                                        'Sigma':params.initial_covariance})
        
        # log prior probability of the dynamics
        QFBb_pri = MNIW(*dynamics_prior_params)
        lp_dyn = QFBb_pri.log_prob({'Matrix':jnp.hstack((params.dynamics_matrix, 
                                                         jnp.hstack((params.dynamics_input_weights,
                                                                     params.dynamics_bias[:,None])))),
                                    'Sigma':params.dynamics_covariance})
        
        # log prior probability of the emission
        RHDd_pri = MNIW(*emission_prior_params)
        lp_ems = RHDd_pri.log_prob({'Matrix':jnp.hstack((params.emission_matrix, 
                                                         jnp.hstack((params.emission_input_weights,
                                                                     params.emission_bias[:,None])))),
                                    'Sigma':params.emission_covariance})
        
        return lp_init + lp_dyn + lp_ems 
    
    def sufficient_stats_from_sample(states):
        """Convert samples of states to sufficient statistics
        
        Returns:
            (initial_states, dynamics_stats, emission_stats)
        """
        # let xn[t] = x[t+1]          for t = 0...T-2
        x, xp, xn = states, states[:-1], states[1:]
        u, up= inputs, inputs[:-1]
        y = emissions
        
        # quantities for the dynamics distribution
        # let zp[t] = [x[t], u[t], 1] for t = 0...T-2
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
        
        return x[0], dynamics_stats, emission_stats
    
    def niw_distribution_update(loc_pri, precision_pri, df_pri, scale_pri, state, N=1):
        """Update the NormalInverseWishart distribution for the initial parameters
        
        Returns:
            the parameters of the posterior NIW distribution
        """
        state = jnp.atleast_2d(state)
        loc_pos = (precision_pri*loc_pri + state.sum(axis=0)) / (precision_pri + N)
        precision_pos = precision_pri + N
        df_pos = df_pri + N
        scale_pos = scale_pri + state.T @ state \
            + precision_pri*jnp.outer(loc_pri, loc_pri) - precision_pos*jnp.outer(loc_pos, loc_pos)
        
        return loc_pos, precision_pos, df_pos, scale_pos
    
    def mniw_distribution_update(M_pri, V_pri, nu_pri, Psi_pri, SxxT, SxyT, SyyT, N):
        """Update the MatrixNormalInverseWishart distribution for the dynamics or the emission

        Args:
            M_pri:   loc of the MNIW prior
            V_pri:   col_precision matrix of the MNIW prior
            nu_pri:  df of the MNIW prior
            Psi_pri: scale matrix of the MNIW prior
            SxxT:    
            SxyT:    
            SyyT:    
            N:       

        Returns:
            the parameters of the posterior MNIW distribution
        """
        Sxx = V_pri + SxxT
        Sxy = SxyT + V_pri @ M_pri.T
        Syy = SyyT + M_pri @ V_pri @ M_pri.T
        M_pos = jnp.linalg.solve(Sxx, Sxy).T
        V_pos = Sxx
        nu_pos = nu_pri + N
        Psi_pos = Psi_pri + Syy - M_pos @ Sxy
        
        return M_pos, V_pos, nu_pos, Psi_pos
        
    def lgssm_params_sample(rng, initial_state, dynamics_stats, emission_stats):
        """Sample parameters of the model.
        """
        rngs = iter(jr.split(rng, 3))
        
        # Sample the initial params
        initial_pos_params = niw_distribution_update(*initial_prior_params, initial_state)
        Sm = NIW(*initial_pos_params).sample(seed=next(rngs))
        S, m = Sm['Sigma'], Sm['mu']
        
        # Sample the dynamics params
        dynamics_pos_params = mniw_distribution_update(*dynamics_prior_params, *dynamics_stats)
        QFBb = MNIW(*dynamics_pos_params).sample(seed=next(rngs))
        Q, FBb = QFBb['Sigma'], QFBb['Matrix']
        F, B, b = FBb[:, :D_hid], FBb[:, D_hid:-1], FBb[:, -1]
        
        # Sample the emission params
        emission_pos_params = mniw_distribution_update(*emission_prior_params, *emission_stats)
        RHDd = MNIW(*emission_pos_params).sample(seed=next(rngs))
        R, HDd = RHDd['Sigma'], RHDd['Matrix']
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
    
    def one_sample(params, rng):
        """One complete iteration of the blocked Gibbs sampler
        """
        rngs = jr.split(rng, 2)
        l_prior = log_prior_prob(params)
        # Sample the states
        ll, states = lgssm_posterior_sample(rngs[0], params, emissions, inputs)
        # Compute sufficient statistics for parameters
        sufficient_stats = sufficient_stats_from_sample(states)
        # Sample parameters
        params_new = lgssm_params_sample(rngs[1], *sufficient_stats)
        log_probs = l_prior + ll
        return params_new, (params_new, log_probs)
    
    rng, *rngs = jr.split(rng, 3+1)
    rngs = iter(rngs)
    
    # Initialize the initial state
    Sm0 = NIW(*initial_prior_params).sample(seed=next(rngs))
    S_0, m_0 = Sm0['Sigma'], Sm0['mu']
    
    # Initialize the dynamics parameters
    QFBb_0 = MNIW(*dynamics_prior_params).sample(seed=next(rngs))
    Q_0, FBb_0 = QFBb_0['Sigma'], QFBb_0['Matrix']
    F_0, B_0, b_0 = FBb_0[:, :D_hid], FBb_0[:, D_hid:-1], FBb_0[:, -1]
    
    # Initialize the emission parameters
    RHDd_0 = MNIW(*emission_prior_params).sample(seed=next(rngs))
    R_0, HDd_0 = RHDd_0['Sigma'], RHDd_0['Matrix']
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
    keys = jr.split(rng, num_itrs)
    _, samples_and_log_probs = lax.scan(one_sample, params_0, keys)
    samples_of_parameters, log_probs = samples_and_log_probs
    
    return samples_of_parameters, log_probs