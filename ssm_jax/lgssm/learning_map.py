import jax.numpy as jnp
import jax.random as jr

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
                 emission_bias=None):
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
        self._prior = None
        
    def initialization_from_prior(cls, key):
        k1, k2, k3 = jr.split(key, num=3)
        m1 = jnp.zeros(state_dim)
        Q1 = jnp.eye(state_dim)
        A = None
        B = None
        b = None
        Q = None
        C = None
        D = None
        d = None
        R = None
        return cls(dynamics_matrix=A,
                   dynamics_covariance=Q,
                   emission_matrix=C,
                   emission_covariance=R,
                   initial_mean=m1,
                   initial_covariance=Q1,
                   dynamics_input_weights=B,
                   dynamics_bias=b,
                   emission_input_weights=D,
                   emission_bias=d)
    
    def marginal_log_evidence(self, emissions, inputs=None):
        """log marginal evidence (log marginal likelihood of observations + log prior of parameters)"""
        mlp = self.marginal_log_prob(emissions, inputs)
        lp = 0
        mle = mlp + lp
        return mle
    
    @classmethod
    def map_step(cls, batch_stats):
        A = None
        Q = None
        C = None
        R = None
        m1 = None
        Q1 = None
        B = None
        b = None
        D = None
        d = None
        return cls(dynamics_matrix=A,
                   dynamics_covariance=Q,
                   emission_matrix=C,
                   emission_covariance=R,
                   initial_mean=m1,
                   initial_covariance=Q1,
                   dynamics_input_weights=B,
                   dynamics_bias=b,
                   emission_input_weights=D,
                   emission_bias=d)
        
    