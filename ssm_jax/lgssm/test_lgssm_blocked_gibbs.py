from jax import random as  jr
from jax import numpy as jnp
from jax import lax

import tensorflow_probability.substrates.jax.distributions as tfd

from blocked_gibbs import lgssm_blocked_gibbs
from ssm_jax.lgssm.models import LinearGaussianSSM


def test_blocked_gibbs(timesteps, seed):
    
    # Set the dimension of the system
    dim_obs = 2
    dim_hid = 4
    dim_in = 6

    # Set true value of the parameter
    Q = 1e-3 * jnp.eye(dim_hid)
    delta = 1.
    F = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = jnp.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]])
    R = 1e-3 * jnp.eye(dim_obs)
    H = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    D = jnp.array([[0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    
    # Set the input
    u1, u2 = 1., 2.
    inputs = jnp.repeat(jnp.array([u1, u1, u1, u1, u2, u2]), timesteps)
    
    # Generate the observation
    keys = jr.split(seed, 3)
    initial_state = jr.multivariate_normal(keys[0], jnp.ones(dim_hid)), Q)
    noise_dynamics = jr.multivariate_normal(keys[1], jnp.zeros(dim_hid), Q, shape=(timesteps-1, ))
    noise_emission = jr.multivariate_normal(keys[2], jnp.zeros(dim_obs), R, shape=(timesteps, ))
    
    def state_update(state, extras):
        input, noise_dyn, noise_ems = extras
        state_new = F @ state[]  + B @ input + noise_dyn
        emission = H @ state_new + D @ input + noise_ems
        return state_new, emission
    
    emissions = lax.scan(state_update, initial_state, zip(inputs, noise_dynamics, noise_emission))    
    
        

# Set the hyperparameter for the prior distribution of parameters
initial_prior_params = 
dynamics_prior_params = 
emission_prior_params = 
prior_hyperparams = (initial_prior_params, dynamics_prior_params, emission_prior_params)

# Run the blocked gibbs sampling algorithm
key = jr.PRNGKey(0)
params_samples, log_probs = lgssm_blocked_gibbs(key, 
                                                num_itrs, 
                                                emissions, 
                                                prior_hyperparams, 
                                                inputs, 
                                                dimension_hidden=None)