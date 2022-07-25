from jax import random as  jr
from jax import numpy as jnp
from jax import lax

from ssm_jax.lgssm.blocked_gibbs import blocked_gibbs


def test_lgssm_blocked_gibbs(num_itrs=10, timesteps=100, seed=jr.PRNGKey(0)):
    
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
    b = 1e-1*jnp.ones(dim_hid)
    R = 1e-3 * jnp.eye(dim_obs)
    H = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    D = jnp.array([[0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    d = 1e-1*jnp.ones(dim_obs)
    
    # Set the input
    u1, u2 = 1., 2.
    inputs = jnp.tile(jnp.array([u1, u1, u1, u1, u2, u2]), (timesteps, 1))
    
    # Generate the observation
    key = iter(jr.split(seed, 4))
    initial_state = jr.multivariate_normal(next(key), jnp.ones(dim_hid), Q)
    noise_dynamics = jr.multivariate_normal(next(key), jnp.zeros(dim_hid), Q, shape=(timesteps-1, ))
    noise_emission = jr.multivariate_normal(next(key), jnp.zeros(dim_obs), R, shape=(timesteps, ))
    
    def state_update(state, extras):
        input, noise_dyn, noise_ems = extras
        state_new = F.dot(state)  + B.dot(input) + b + noise_dyn
        emission = H.dot(state_new) + D.dot(input) + d + noise_ems
        return state_new, emission
    
    emission_1st = H.dot(initial_state) + D.dot(inputs[0]) + d + noise_emission[0]
    _, _emissions = lax.scan(state_update, initial_state, (inputs[1:], noise_dynamics, noise_emission[1:])) 
    emissions = jnp.row_stack((emission_1st, _emissions))   
    
    # Set the hyperparameter for the prior distribution of parameters
    loc_init, precision_init, df_init, scale_init = jnp.ones(dim_hid), 1., dim_hid, jnp.eye(dim_hid)
    initial_prior_params = (loc_init, precision_init, df_init, scale_init)
    
    M_dyn, V_dyn = jnp.hstack((F, B, b[:,None])), jnp.eye(dim_hid+dim_in+1)
    nu_dyn, Psi_dyn = dim_hid, jnp.eye(dim_hid)
    dynamics_prior_params = (M_dyn, V_dyn, nu_dyn, Psi_dyn)
    
    M_ems, V_ems = jnp.hstack((H, D, d[:,None])), jnp.eye(dim_hid+dim_in+1)
    nu_ems, Psi_ems = dim_obs, jnp.eye(dim_obs)
    emission_prior_params = (M_ems, V_ems, nu_ems, Psi_ems)
    
    prior_hyperparams = (initial_prior_params, dynamics_prior_params, emission_prior_params)

    # Run the blocked gibbs sampling algorithm
    samples_of_parameters, log_probs = blocked_gibbs(jr.PRNGKey(0), 
                                                     num_itrs, 
                                                     emissions, 
                                                     prior_hyperparams, 
                                                     inputs, 
                                                     D_hid=None)
    
    return samples_of_parameters, log_probs