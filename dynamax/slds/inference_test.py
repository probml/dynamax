"""
Tests for the inference functions in dynamax/slds/inference.py
"""
import pytest 

import jax.numpy as jnp
import jax.random as jr
import dynamax.slds.mixture_kalman_filter_demo as kflib
import jax

from dynamax.slds import SLDS, DiscreteParamsSLDS, LGParamsSLDS, ParamsSLDS, rbpfilter, rbpfilter_optimal
from functools import partial
from functools import partial
from jax.scipy.special import logit


class TestRBPF():
    """
    Tests for the inference functions in dynamax/slds/inference.py
    """
    ## Model definitions
    num_states = 3
    num_particles = 10
    state_dim = 4
    emission_dim = 4

    TT = 1
    A = jnp.array([[1, TT, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, TT],
                [0, 0, 0, 1]])


    B1 = jnp.array([0, 0, 0, 0])
    B2 = jnp.array([-1.225, -0.35, 1.225, 0.35])
    B3 = jnp.array([1.225, 0.35,  -1.225,  -0.35])
    B = jnp.stack([B1, B2, B3], axis=0)

    Q = 0.2 * jnp.eye(4)
    R = 10.0 * jnp.diag(jnp.array([2, 1, 2, 1]))
    C = jnp.eye(4)

    transition_matrix = jnp.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])

    discr_params = DiscreteParamsSLDS(
        initial_distribution=jnp.ones(num_states)/num_states,
        transition_matrix=transition_matrix,
        proposal_transition_matrix=transition_matrix
    )

    lg_params = LGParamsSLDS(
        initial_mean=jnp.ones(state_dim),
        initial_cov=jnp.eye(state_dim),
        dynamics_weights=A,
        dynamics_cov=Q,
        dynamics_bias=jnp.array([B1, B2, B3]),
        dynamics_input_weights=None,
        emission_weights=C,
        emission_cov=R,
        emission_bias=None,
        emission_input_weights=None
    )

    pre_params = ParamsSLDS(
        discrete=discr_params,
        linear_gaussian=lg_params
    )

    params = pre_params.initialize(num_states, state_dim, emission_dim)

    ## Sample states and emissions
    key = jr.PRNGKey(1)
    slds = SLDS(num_states, state_dim, emission_dim)
    dstates, cstates, emissions = slds.sample(params, key, 100)

    ## Baseline Implementation parameters          
    key_base = jr.PRNGKey(31)
    key_mean_init, key_sample, key_state, key_next = jr.split(key_base, 4)
    p_init = jnp.array([0.0, 1.0, 0.0])

    mu_0 = 0.01 * jr.normal(key_mean_init, (num_particles, 4))
    Sigma_0 = jnp.zeros((num_particles, 4,4))
    s0 = jr.categorical(key_state, logit(p_init), shape=(num_particles,))
    weights_0 = jnp.ones(num_particles) / num_particles
    init_config = (key_next, mu_0, Sigma_0, weights_0, s0)
    params1 = kflib.RBPFParamsDiscrete(A, B, C, Q, R, transition_matrix)

    def test_rbpf(self):
        """ 
        Test the RBPF implementation
        """
        # Baseline
        rbpf_optimal_part = partial(kflib.rbpf, params=self.params1, nparticles=self.num_particles)
        _, (mu_hist, Sigma_hist, weights_hist, s_hist, Ptk) = jax.lax.scan(rbpf_optimal_part, self.init_config, self.emissions)
        bl_post_mean = jnp.einsum("ts,tsm->tm", weights_hist, mu_hist)

        bl_rbpf_mse = ((bl_post_mean - self.cstates)[:, [0, 2]] ** 2).mean(axis=0).sum()
        # Dynamax
        out = rbpfilter(self.num_particles, self.params, self.emissions, self.key)
        means = out['means']
        weights = out['weights']
        dyn_post_mean = jnp.einsum("ts,tsm->tm", weights, means)
        dyn_rbpf_mse = ((dyn_post_mean - self.cstates)[:, [0, 2]] ** 2).mean(axis=0).sum()
        print(bl_rbpf_mse, dyn_rbpf_mse)
        assert jnp.allclose(bl_post_mean, dyn_post_mean, atol=10.0)

    def test_rbpf_optimal(self):
        """
        Test the RBPF optimal implementation
        """
        # Baseline
        rbpf_optimal_part = partial(kflib.rbpf_optimal, params=self.params1, nparticles=self.num_particles)
        _, (mu_hist, Sigma_hist, weights_hist, s_hist, Ptk) = jax.lax.scan(rbpf_optimal_part, self.init_config, self.emissions)
        bl_post_mean = jnp.einsum("ts,tsm->tm", weights_hist, mu_hist)
        bl_rbpf_mse = ((bl_post_mean - self.cstates)[:, [0, 2]] ** 2).mean(axis=0).sum()
        latent_hist_est = Ptk.mean(axis=1).argmax(axis=1)
        # Dynamax
        out = rbpfilter_optimal(self.num_particles, self.params, self.emissions, self.key)
        means = out['means']
        weights = out['weights']
        dyn_post_mean = jnp.einsum("ts,tsm->tm", weights, means)
        dyn_rbpf_mse = ((dyn_post_mean - self.cstates)[:, [0, 2]] ** 2).mean(axis=0).sum()
        print(bl_rbpf_mse, dyn_rbpf_mse)
        assert jnp.allclose(bl_post_mean, dyn_post_mean, atol=10.0)
    



