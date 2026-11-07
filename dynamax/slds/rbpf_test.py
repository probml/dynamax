"""
Tests for the Rao-Blackwellized particle filters in dynamax/slds/rbpf.py
"""
import pytest 

import jax.numpy as jnp
import jax.random as jr
import dynamax.slds.mixture_kalman_filter_demo as kflib
import jax

from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_joint_sample, make_lgssm_params
from dynamax.slds import DiscreteParamsRBPF, LGParamsRBPF, ParamsRBPF, rbpfilter, rbpfilter_optimal
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from functools import partial
from jax.scipy.special import logit


def sample_rbpf_model(params, key, num_timesteps):
    """Sample (z, x, y) from the SLDS that `ParamsRBPF` describes: z_0 ~ pi, x_0 ~ N(mu_{z_0}, Sigma_{z_0}),
    then transitions and emissions (no inputs)."""
    lg, num_states = params.linear_gaussian, params.discrete.initial_distribution.shape[0]

    def _step(prev_states, key):
        key0, key1, key2 = jr.split(key, 3)
        dstate, cstate = prev_states
        dstate = jr.choice(key0, jnp.arange(num_states), p=params.discrete.transition_matrix[dstate, :])
        cstate = MVN(lg.dynamics_weights[dstate] @ cstate + lg.dynamics_bias[dstate], lg.dynamics_cov[dstate]).sample(seed=key2)
        emission = MVN(lg.emission_weights[dstate] @ cstate + lg.emission_bias[dstate], lg.emission_cov[dstate]).sample(seed=key1)
        return (dstate, cstate), (dstate, cstate, emission)

    key0, key1, key2, key = jr.split(key, 4)
    z0 = jr.choice(key0, jnp.arange(num_states), p=params.discrete.initial_distribution)
    x0 = MVN(lg.initial_mean[z0], lg.initial_cov[z0]).sample(seed=key1)
    y0 = MVN(lg.emission_weights[z0] @ x0 + lg.emission_bias[z0], lg.emission_cov[z0]).sample(seed=key2)
    _, (zs, xs, ys) = lax.scan(_step, (z0, x0), jr.split(key, num_timesteps - 1))
    cat = lambda a0, a: jnp.concatenate((a0[None], a))
    return cat(z0, zs), cat(x0, xs), cat(y0, ys)


class TestRBPF():
    """
    Tests for the inference functions in dynamax/slds/rbpf.py
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

    # A proposal different from the transition matrix, so rbpfilter's importance ratio is exercised
    proposal_transition_matrix = jnp.ones((num_states, num_states)) / num_states

    discr_params = DiscreteParamsRBPF(
        initial_distribution=jnp.ones(num_states)/num_states,
        transition_matrix=transition_matrix,
        proposal_transition_matrix=proposal_transition_matrix
    )

    lg_params = LGParamsRBPF(
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

    pre_params = ParamsRBPF(
        discrete=discr_params,
        linear_gaussian=lg_params
    )

    params = pre_params.initialize(num_states, state_dim, emission_dim)

    ## Sample states and emissions
    num_timesteps = 100
    key = jr.PRNGKey(1)
    dstates, cstates, emissions = sample_rbpf_model(params, key, num_timesteps)

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

    # The baseline filters a different model (it starts at z = 1, x ~ 0 with zero covariance, and
    # samples z_t with logits logit(p) rather than log(p)), so the posterior means differ most in
    # the first steps. Measured max abs difference: 6.25 (rbpfilter) and 3.62 (rbpfilter_optimal);
    # at most 6.9 and 5.7 over 20 filter keys.
    atol = 8.0

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
        assert jnp.allclose(bl_post_mean, dyn_post_mean, atol=self.atol)

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
        assert jnp.allclose(bl_post_mean, dyn_post_mean, atol=self.atol)

    @pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
    def test_jit(self, filter_fn):
        """Both filters run under jax.jit with num_particles static."""
        out = jax.jit(filter_fn, static_argnums=0)(self.num_particles, self.params, self.emissions, self.key)
        T, P, D = self.num_timesteps, self.num_particles, self.state_dim
        assert out["weights"].shape == (T, P)
        assert out["states"].shape == (T, P)
        assert out["means"].shape == (T, P, D)
        assert out["covariances"].shape == (T, P, D, D)
        assert jnp.all(jnp.isfinite(out["means"])) and jnp.all(jnp.isfinite(out["covariances"]))
        assert jnp.allclose(out["weights"].sum(axis=1), 1.0)


@pytest.mark.parametrize("filter_fn", [rbpfilter, rbpfilter_optimal])
def test_single_state_matches_kalman_filter(filter_fn):
    """With one discrete state every particle runs the same Kalman filter, whatever the weights, so each
    particle's filtered moments equal lgssm_filter's. This checks that x_0 starts at N(mu, Sigma) at time 0
    (not sampled, and with no transition before y_0) and the input indexing."""
    state_dim, emission_dim, input_dim, num_timesteps, num_particles = 3, 2, 2, 50, 7
    keys = jr.split(jr.PRNGKey(0), 13)
    random_psd = lambda key, dim: (lambda L: L @ L.T + 0.1 * jnp.eye(dim))(jr.normal(key, (dim, dim)) / jnp.sqrt(dim))
    initial_mean = jr.normal(keys[0], (state_dim,))
    initial_cov = random_psd(keys[1], state_dim)
    dynamics_weights = 0.8 * jnp.eye(state_dim) + 0.1 * jr.normal(keys[2], (state_dim, state_dim))
    dynamics_cov = random_psd(keys[3], state_dim)
    dynamics_bias = jr.normal(keys[4], (state_dim,))
    dynamics_input_weights = jr.normal(keys[5], (state_dim, input_dim))
    emission_weights = jr.normal(keys[6], (emission_dim, state_dim))
    emission_cov = random_psd(keys[7], emission_dim)
    emission_bias = jr.normal(keys[8], (emission_dim,))
    emission_input_weights = jr.normal(keys[9], (emission_dim, input_dim))

    lgssm_params = make_lgssm_params(initial_mean, initial_cov, dynamics_weights, dynamics_cov,
                                     emission_weights, emission_cov, dynamics_bias, dynamics_input_weights,
                                     emission_bias, emission_input_weights)
    rbpf_params = ParamsRBPF(
        discrete=DiscreteParamsRBPF(
            initial_distribution=jnp.ones(1),
            transition_matrix=jnp.ones((1, 1)),
            proposal_transition_matrix=jnp.ones((1, 1))),
        linear_gaussian=LGParamsRBPF(
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            dynamics_weights=dynamics_weights,
            dynamics_cov=dynamics_cov,
            dynamics_bias=dynamics_bias,
            dynamics_input_weights=dynamics_input_weights,
            emission_weights=emission_weights,
            emission_cov=emission_cov,
            emission_bias=emission_bias,
            emission_input_weights=emission_input_weights)
    ).initialize(1, state_dim, emission_dim, input_dim)

    inputs = jr.normal(keys[10], (num_timesteps, input_dim))
    _, emissions = lgssm_joint_sample(lgssm_params, keys[11], num_timesteps, inputs)
    expected = lgssm_filter(lgssm_params, emissions, inputs)
    out = filter_fn(num_particles, rbpf_params, emissions, keys[12], inputs=inputs)

    assert jnp.allclose(out["means"], expected.filtered_means[:, None], atol=1e-5)
    assert jnp.allclose(out["covariances"], expected.filtered_covariances[:, None], atol=1e-5)
