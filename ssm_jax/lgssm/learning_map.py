import jax.numpy as jnp
import jax.random as jr
from jax import jit
from distrax import MultivariateNormalFullCovariance as MVN
from NIW import InverseWishart as IW
from _ import MatrixNormal as MN

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
                 prior=None):
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
        self._prior = prior
    
    @classmethod
    def initialization_from_prior(cls, rng, prior):
        rngs = iter(jr.split(rng, 8))
        # Initialize the initial state
        S = IW().sample(seed=next(rngs))
        m = MVN(prior.m_mu, prior.m_cov).sample(seed=next(rngs))
        # Initialize the dynamics parameters
        Q = IW().sample(seed=next(rngs))
        F = MN().sample(seed=next(rngs)) 
        Bb = MN().sample(seed=next(rngs)) 
        b = Bb[:,0]
        B = Bb[:,1:]
        # Initialize the emission parameters
        R = IW().sample(seed=next(rngs))
        H = MN().sample(seed=next(rngs)) 
        Dd = MN().sample(seed=next(rngs))
        D = Dd[:,0]
        d = Dd[:,1:]
        return cls(dynamics_matrix=F,
                   dynamics_covariance=Q,
                   emission_matrix=H,
                   emission_covariance=R,
                   initial_mean=m,
                   initial_covariance=S,
                   dynamics_input_weights=B,
                   dynamics_bias=b,
                   emission_input_weights=D,
                   emission_bias=d)
    
    def marginal_log_evidence(self, emissions, inputs=None):
        """log marginal evidence (log marginal likelihood of observations + log prior of parameters)"""
        mlp = self.marginal_log_prob(emissions, inputs)
        lp = self._prior.log_prob()
        mle = mlp + lp
        return mle
    
    @classmethod
    def map_step(cls, batch_stats):
        F = None
        Q = None
        H = None
        R = None
        m = None
        S = None
        B = None
        b = None
        D = None
        d = None
        return cls(dynamics_matrix=F,
                   dynamics_covariance=Q,
                   emission_matrix=H,
                   emission_covariance=R,
                   initial_mean=m,
                   initial_covariance=S,
                   dynamics_input_weights=B,
                   dynamics_bias=b,
                   emission_input_weights=D,
                   emission_bias=d)
        

def lgssm_fit_emap(model, batch_emission, num_iters=50):

    @jit
    def emap_step(model):
        posterior_stats, marginal_loglikes = model.e_step(batch_emission)
        model = model.map_step(posterior_stats)
        marginal_log_evs = 0
        return model, marginal_log_evs.sum()
    
    log_evs = []
    for _ in trange(num_iters):
        model, marginal_evidence = emap_step(model)
        log_evs.append(marginal_evidence)
        
    return model, jnp.array(log_evs)