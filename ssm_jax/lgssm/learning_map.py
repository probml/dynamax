from functools import partial
from ssl import VERIFY_X509_PARTIAL_CHAIN

import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.tree_util import register_pytree_node_class, tree_map
from distrax import MultivariateNormalFullCovariance as MVN

import tensorflow_probability.substrates.jax.distributions as tfd
MN = tfd.MatrixNormalLinearOperator

from NIW import InverseWishart as IW 

from tqdm.auto import trange

from models import LinearGaussianSSM

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
                 prior_params=None):
        super().__init__(self,
                         dynamics_matrix,
                         dynamics_covariance,
                         emission_matrix,
                         emission_covariance,
                         initial_mean,
                         initial_covariance,
                         dynamics_input_weights,
                         dynamics_bias,
                         emission_input_weights,
                         emission_bias)
        self.initial_prior_params = prior_params['initial']
        self.dynamics_prior_params = prior_params['dynamics']
        self.emission_prior_params = prior_params['emission']
    
    def m_step(self, batch_stats, MAP=True):
        
        def fit_linear_regression(ExxT, ExyT, EyyT, L):
            # Solve a linear regression given sufficient statistics
            W = jnp.linalg.solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / L
            return W, Sigma
        
        def fit_penalized_regression(ExxT, ExyT, EyyT, L, M_pri, V_pri, nu_pri, Psi_pri):
            Sxx = ExxT + V_pri 
            Sxy = ExyT + V_pri @ M_pri.T
            Syy = EyyT + M_pri @ Sxy
            W = jnp.linalg.solve(Sxx, Sxy).T
            Sigma = Psi_pri + Syy - Sxy.T @ jnp.linalg.solve(Sxx, Sxy)
            Sigma = Sigma / (nu_pri + L + 2*dim + 1)
            return W, Sigma            
        
        # Sum the statistics across all batches 
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats
        sum_x0, sum_x0x0T, N = init_stats
        dim = sum_x0.shape[0]
        
        if MAP:
            # initial distribution 
            loc_pri, precision_pri, df_pri, scale_pri = self.initial_prior_params
            
            m = (precision_pri * loc_pri + sum_x0) / (precision_pri + N)
            S = scale_pri + (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) \
                - jnp.outer(sum_x0/N-loc_pri, sum_x0/N-loc_pri)*(precision_pri*N)/(precision_pri + N)
            S /= df_pri + N + dim + 2
            
            # dynamics distribution
            W_d, Q = fit_penalized_regression(*dynamics_stats, *self.dynamics_prior_params)
            F, B, b = W_d[:, :dim], W_d[:, dim:-1], W_d[:, -1]
            
            # emission distribution
            W_d, Q = fit_penalized_regression(*dynamics_stats, *self.emission_prior_params)
            H, D, d = W_e[:, :dim], W_e[:, dim:-1], W_e[:, -1]
            
        else:
            # initial distribution
            m = sum_x0 / N
            S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
            
            # dynamics distribution
            W_d, Q = fit_linear_regression(*dynamics_stats)
            F, B, b = W_d[:, :dim], W_d[:, dim:-1], W_d[:, -1]
            
            # emission distribution
            W_e, R = fit_linear_regression(*emission_stats)
            H, D, d = W_e[:, :dim], W_e[:, dim:-1], W_e[:, -1]
            
        hmm = _LinearGaussianSSM(dynamics_matrix=F,
                                 dynamics_covariance=Q,
                                 emission_matrix=H,
                                 emission_covariance=R,
                                 initial_mean=m,
                                 initial_covariance=S,
                                 dynamics_input_weights=B,
                                 dynamics_bias=b,
                                 emission_input_weights=D,
                                 emission_bias=d)
        return hmm, 
    