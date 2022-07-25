from functools import partial

import jax.numpy as jnp
from jax.tree_util import tree_map

from ssm_jax.lgssm.models import LinearGaussianSSM
from ssm_jax.lgssm.blocked_gibbs import niw_distribution_update, mniw_distribution_update
from ssm_jax.utils_distributions import NormalInverseWishart as NIW, MatrixNormalInverseWishart as MNIW

class _LinearGaussianSSM(LinearGaussianSSM):
    def __init__(self,
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
                 prior_hyperparams=None):
        super().__init__(dynamics_matrix,
                         dynamics_covariance,
                         emission_matrix,
                         emission_covariance,
                         initial_mean,
                         initial_covariance,
                         dynamics_input_weights,
                         dynamics_bias,
                         emission_input_weights,
                         emission_bias)
        if prior_hyperparams is not None:
            self.initial_prior_params = prior_hyperparams[0]
            self.dynamics_prior_params = prior_hyperparams[1]
            self.emission_prior_params = prior_hyperparams[2]
        else:
            # Set hyperparameters of the prior
            # hyperparameters of the NormalInverseWishart prior for initial state
            _niw_params = jnp.zeros(self.state_dim), 1., self.state_dim, 1e4*jnp.eye(self.state_dim)
            self.initial_prior_params = _niw_params
            # hyperparameters of MatrixNormalInverseWishart prior for dynamics 
            _mniw_params = (jnp.zeros((self.state_dim, self.state_dim + self.input_dim + 1)), 
                            jnp.eye(self.state_dim + self.input_dim + 1), 
                            self.state_dim, 
                            1e4 * jnp.eye(self.state_dim))
            self.dynamics_prior_params = _mniw_params
            _mniw_params = (jnp.zeros((self.emission_dim, self.state_dim + self.input_dim + 1)), 
                            jnp.eye(self.state_dim + self.input_dim + 1), 
                            self.emission_dim, 
                            1e4 * jnp.eye(self.emission_dim))
            self.emission_prior_params = _mniw_params
    
    def m_step(self, batch_stats):
        # Sum the statistics across all batches 
        _stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = _stats
        
        # Initial posterior distribution 
        initial_pos_params = niw_distribution_update(*self.initial_prior_params, *init_stats)
        m, S = NIW(*initial_pos_params).mode
        # Dynamics posterior distribution
        dynamics_pos_params = mniw_distribution_update(*self.dynamics_prior_params, *dynamics_stats)
        FBb, Q = MNIW(*dynamics_pos_params).mode
        F, B, b = FBb[:, :self.state_dim], FBb[:, self.state_dim:-1], FBb[:, -1]
        # Emission posterior distribution
        emission_pos_params = mniw_distribution_update(*self.emission_prior_params, *emission_stats)
        HDd, R = MNIW(*emission_pos_params).mode
        H, D, d = HDd[:, :self.state_dim], HDd[:, self.state_dim:-1], HDd[:, -1]
            
        return  _LinearGaussianSSM(dynamics_matrix=F,
                                   dynamics_covariance=Q,
                                   emission_matrix=H,
                                   emission_covariance=R,
                                   initial_mean=m,
                                   initial_covariance=S,
                                   dynamics_input_weights=B,
                                   dynamics_bias=b,
                                   emission_input_weights=D,
                                   emission_bias=d,
                                   prior_hyperparams=self.prior_hyperparams)
        
        
#TODO: batch m step? --- update posterior with a batch of observation?

