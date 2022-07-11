import jax.numpy as jnp
import jax.random as jr
from jax import lax
import chex
from distrax import MultivariateNormalFullCovariance as MVN
from inference import LGSSMParams, lgssm_posterior_sample

from _ import MatrixNormal as MN
from NIW import InverseWishart as IW


@chex.dataclass
class LGSSMParamsDistribution:
    """
    m (D_hid,): intial_mean
    S (D_hid, D_hid): initial_covariance
    F (D_hid, D_hid): dynamics_matrix
    B (D_hid, D_in): dynamics_input_weights
    b (D_hid,): dynamics_bias
    Q (D_hid, D_hid): dynamics_covariance
    H (D_obs, D_hid): emission_matrix
    D (D_obs, D_in): emission_input_weights
    d (D_obs,): emission_bias
    R (D_obs, D_obs): emission_covariance
    """
    # intial_mean follows MVN distribution
    m_mu: chex.Array
    m_cov: chex.Array
    # initial_covariance
    
    # dynamics_matrix
    F_mu:chex.Array
    # dynamics_input_weights
    # dynamics_bias
    # dynamics_covariance
    # emission_matrix
    H_mu: chex.Array
    # emission_input_weights
    # emission_bias
    # emission_covariance
    
    def log_probability(self, params):
        lp_m = MVN(self.m_mu, self.m_cov).log_prob(params.initial_mean)
        lp_S = IW().log_prob(params.initial_covariance)
        lp_F = MN().log_prob(params.dynamics_matrix)
        lp_Bb = MN().log_prob(jnp.vstack((params.dynamics_bias, params.dynamics_input_weights)))
        lp_Q = IW().log_prob(params.dynamics_covariance)
        lp_H = MN().log_prob(params.emission_matrix)
        lp_Dd = MN().log_prob(jnp.vstack((params.emission_bias, params.emission_input_weights)))
        lp_R = IW().log_prob(params.emission_covariance)
        return lp_m + lp_S + lp_F + lp_Bb + lp_Q + lp_H + lp_Dd + lp_R


def lgssm_blocked_gibbs(rng, num_itrs, emissions, prior=None, inputs=None, dimension_hidden=None):
    """Estimation using blocked-Gibbs sampler
    
    Assume that parameters are fixed over time

    Args:
        rng (_type_): _description_
        emissions (_type_): _description_
        prior (_type_): _description_
        inputs (_type_, optional): _description_. Defaults to None.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    
    def _lgssm_params_sample(rng, params_current, states, inputs, emissions):
        m = params_current.initial_mean
        S = params_current.initial_covariance
        F = params_current.dynamics_matrix
        B = params_current.dynamics_input_weights
        b = params_current.dynamics_bias
        Q = params_current.dynamics_covariance
        H = params_current.emission_matrix
        D = params_current.emission_input_weights
        d = params_current.emission_bias
        R = params_current.emission_covariance
        
        rngs = iter(jr.split(rng, 8))
        # Sample the initial params
        S_posterior = None
        S = IW().sample(seed=next(rngs))
        m_posterior = None
        m = MVN().sample(seed=next(rngs))
        
        # Sample the dynamics params
        Q_posterior = None
        Q = IW().sample(seed=next(rngs))
        F_posterior = None
        F = MN().sample(seed=next(rngs))
        Bb_posterior = None
        Bb = MN().sample(seed=next(rngs))
        b = Bb[:,0]
        B = Bb[:,1:]
        
        # Sample the emission params
        R_posterior = None
        R = IW().sample(seed=next(rngs))
        H_posterior = None
        H = MN().sample(seed=next(rngs))
        Dd_posterior = None
        Dd = MN.sample(seed=next(rngs))
        d = Dd[:,0]
        D = Dd[:,1:]
        
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
    
    def _one_sample(params_current, rng):
        rngs = jr.split(rng, 2)
        ll, states = lgssm_posterior_sample(rngs[0], params_current, emissions, inputs)
        params_new = _lgssm_params_sample(rngs[1], params_current, states, inputs, emissions)
        l_prior = prior.log_probability(params_new)
        log_evidence = ll + l_prior
        return (params_new, log_evidence), params_new
    
    # Initialize the parameters from the prior
    if prior is None:
        D_hid = dimension_hidden
        D_obs = emissions.shape[1]
        D_in = inputs.shapes[1]
        prior = LGSSMParamsDistribution(m_mu = jnp.zeros(D_hid), m_cov = jnp.eye(D_hid)) 
    
    rng, *rngs = jr.split(rng, 8+1)
    rngs = iter(rngs)
    # Initialize the initial state
    S_0 = IW().sample(seed=next(rngs))
    m_0 = MVN(prior.m_mu, prior.m_cov).sample(seed=next(rngs))
    # Initialize the dynamics parameters
    Q_0 = IW().sample(seed=next(rngs))
    F_0 = MN().sample(seed=next(rngs)) 
    Bb_0 = MN().sample(seed=next(rngs)) 
    b_0 = Bb_0[:,0]
    B_0 = Bb_0[:,1:]
    # Initialize the emission parameters
    R_0 = IW().sample(seed=next(rngs))
    H_0 = MN().sample(seed=next(rngs)) 
    Dd_0 = MN().sample(seed=next(rngs))
    D_0 = Dd_0[:,0]
    d_0 = Dd_0[:,1:]
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
    params_samples, log_evidence = lax.scan(_one_sample, params_0, rngs)
    
    return params_samples, log_evidence