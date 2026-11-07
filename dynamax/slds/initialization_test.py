"""
Tests of the PCA + ARHMM initialization.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from dynamax.slds import StationarySLDSTransitions, GaussianSLDS, ParamsSLDS, arhmm_initialization, pca_initialization
from dynamax.slds.initialization import MIN_NOISE_VARIANCE, _ARHMM, _ParamsARHMM
from dynamax.utils.utils import find_permutation


def principal_angle_cosines(A, B):
    """Cosines of the principal angles between col(A) and col(B)."""
    return jnp.linalg.svd(jnp.linalg.qr(A)[0].T @ jnp.linalg.qr(B)[0])[1]


def linear_subspace_data(key, num_sequences=2, num_timesteps=300, state_dim=2, emission_dim=6, input_dim=0, noise=0.0):
    """y = H x + d + D u + noise with a random H; returns (x, u, y, H, d, D)."""
    k1, k2, k3, k4, k5 = jr.split(key, 5)
    H = jr.normal(k1, (emission_dim, state_dim))
    d = jr.normal(k2, (emission_dim,))
    Dm = jr.normal(k3, (emission_dim, input_dim))
    x = jr.normal(k4, (num_sequences, num_timesteps, state_dim))
    u = jr.normal(k5, (num_sequences, num_timesteps, input_dim))
    y = x @ H.T + d + u @ Dm.T + noise * jr.normal(jr.PRNGKey(99), (num_sequences, num_timesteps, emission_dim))
    return x, u, y, H, d, Dm


class TestPCA:
    def test_recovers_subspace_noiselessly(self):
        x, _, y, H, d, _ = linear_subspace_data(jr.PRNGKey(0))
        params, x_hat = pca_initialization(jr.PRNGKey(1), y, 2)
        assert params.weights.shape == (1, 6, 2) and x_hat.shape == x.shape
        assert jnp.allclose(principal_angle_cosines(params.weights[0], H), 1.0, atol=1e-4)
        assert jnp.allclose(params.biases[0], d + x.reshape(-1, 2).mean(0) @ H.T, atol=1e-4)
        # x̂ equals x up to an invertible linear map and offset: regress x on [x̂, 1]
        design = jnp.concatenate([x_hat.reshape(-1, 2), jnp.ones((x_hat.shape[0] * x_hat.shape[1], 1))], 1)
        coef = jnp.linalg.lstsq(design, x.reshape(-1, 2))[0]
        assert jnp.allclose(design @ coef, x.reshape(-1, 2), atol=1e-3)
        # noiseless data: the noise variance is clamped
        assert jnp.allclose(params.covs, MIN_NOISE_VARIANCE)

    def test_noise_variance_and_full_cov(self):
        _, _, y, _, _, _ = linear_subspace_data(jr.PRNGKey(0), num_timesteps=2000, emission_dim=8, noise=0.3)
        params, _ = pca_initialization(jr.PRNGKey(1), y, 2, cov_type="full")
        assert params.covs.shape == (1, 8, 8)
        assert jnp.allclose(jnp.diag(params.covs[0]), 0.09, rtol=0.1)
        assert jnp.allclose(params.covs[0] - jnp.diag(jnp.diag(params.covs[0])), 0.0)

    def test_inputs(self):
        x, u, y, H, d, Dm = linear_subspace_data(jr.PRNGKey(0), num_timesteps=2000, input_dim=3)
        params, x_hat = pca_initialization(jr.PRNGKey(1), y, 2, inputs=u)
        assert params.input_weights.shape == (1, 6, 3)
        assert jnp.allclose(params.input_weights[0], Dm, atol=0.1)
        assert jnp.allclose(principal_angle_cosines(params.weights[0], H), 1.0, atol=1e-3)

    def test_multiple_subspaces_are_distinct(self):
        _, _, y, _, _, _ = linear_subspace_data(jr.PRNGKey(0), state_dim=2, emission_dim=10, noise=0.1)
        params, x_hat = pca_initialization(jr.PRNGKey(1), y, 2, num_subspaces=3)
        assert params.weights.shape == (3, 10, 2) and x_hat.shape[-1] == 2
        for k in range(3):
            assert jnp.linalg.matrix_rank(params.weights[k]) == 2
            for j in range(k):
                assert principal_angle_cosines(params.weights[k], params.weights[j]).min() < 0.99

    def test_emission_dim_equal_to_state_dim(self):
        _, _, y, _, _, _ = linear_subspace_data(jr.PRNGKey(0), state_dim=3, emission_dim=3, noise=0.1)
        params, x_hat = pca_initialization(jr.PRNGKey(1), y, 3)
        assert params.weights.shape == (1, 3, 3) and jnp.all(jnp.isfinite(x_hat))
        assert jnp.allclose(params.covs, MIN_NOISE_VARIANCE)

    def test_rejects_too_small_emission_dim(self):
        _, _, y, _, _, _ = linear_subspace_data(jr.PRNGKey(0), state_dim=2, emission_dim=3)
        with pytest.raises(ValueError):
            pca_initialization(jr.PRNGKey(1), y, 4)


def switching_latents(key, num_sequences=1, num_timesteps=500, input_dim=0):
    """x from a K=2 SLDS whose states rotate in opposite directions around different points."""
    model = GaussianSLDS(2, 2, 4, input_dim)
    params, _ = model.initialize(jr.PRNGKey(0))
    rot = lambda s: 0.98 * jnp.array([[jnp.cos(s), -jnp.sin(s)], [jnp.sin(s), jnp.cos(s)]])
    params = params._replace(
        transitions=params.transitions._replace(transition_matrix=jnp.array([[0.97, 0.03], [0.03, 0.97]])),
        dynamics=params.dynamics._replace(weights=jnp.stack([rot(0.5), rot(-0.5)]), biases=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
                                          input_weights=0.3 * jnp.ones((2, 2, input_dim)), covs=jnp.tile(0.05 * jnp.eye(2), (2, 1, 1))))
    inputs = jr.normal(jr.fold_in(key, 1), (num_sequences, num_timesteps, input_dim))
    z, x, _ = vmap(lambda k, u: model.sample(params, k, num_timesteps, u))(jr.split(key, num_sequences), inputs)
    return model, params, z, x, inputs


class TestARHMM:
    def test_recovers_states_and_dynamics_from_true_latents(self):
        model, true_params, z, x, inputs = switching_latents(jr.PRNGKey(0))
        initial, transitions, dynamics, score = arhmm_initialization(jr.PRNGKey(1), model, x, inputs, num_iters=30)
        assert jnp.isfinite(score)
        arhmm = _ARHMM(model)
        params = _ParamsARHMM(initial, transitions, dynamics)
        z_hat = arhmm.most_likely_states(params, x[0], jnp.concatenate([x[0], inputs[0]], -1))
        perm = find_permutation(z_hat, z[0])
        assert jnp.mean(perm[z_hat] == z[0]) > 0.9
        for k in range(2):
            assert jnp.allclose(dynamics.weights[k], true_params.dynamics.weights[perm[k]], atol=0.1)
            assert jnp.allclose(dynamics.biases[k], true_params.dynamics.biases[perm[k]], atol=0.15)
        assert jnp.allclose(transitions.transition_matrix.sum(1), 1.0, atol=1e-5)
        assert jnp.all(jnp.diag(transitions.transition_matrix) > 0.8)
        # the initial distribution of x stays frozen at (0, I)
        assert jnp.array_equal(dynamics.initial_mean, jnp.zeros(2)) and jnp.array_equal(dynamics.initial_cov, jnp.eye(2))

    def test_inputs_are_fitted(self):
        model, true_params, z, x, inputs = switching_latents(jr.PRNGKey(2), input_dim=2)
        _, _, dynamics, _ = arhmm_initialization(jr.PRNGKey(1), model, x, inputs, num_iters=30)
        assert dynamics.input_weights.shape == (2, 2, 2)
        assert jnp.allclose(dynamics.input_weights, 0.3, atol=0.1)

    def test_pooled_sequences_and_restarts(self):
        model, _, _, x, inputs = switching_latents(jr.PRNGKey(3), num_sequences=3, num_timesteps=100)
        single = arhmm_initialization(jr.PRNGKey(1), model, x, inputs, num_iters=10, num_restarts=1)
        several = arhmm_initialization(jr.PRNGKey(1), model, x, inputs, num_iters=10, num_restarts=3)
        assert single[2].weights.shape == (2, 2, 2)
        assert several[3] >= single[3] - 1e-3 * jnp.abs(single[3])

    def test_zero_iterations_gives_kmeans_start(self):
        model, _, _, x, inputs = switching_latents(jr.PRNGKey(4), num_timesteps=100)
        initial, transitions, dynamics, score = arhmm_initialization(jr.PRNGKey(1), model, x, inputs, num_iters=0)
        assert jnp.isfinite(score) and jnp.all(jnp.isfinite(dynamics.weights))
        assert jnp.allclose(initial.probs, 0.5)


class TestModelInitialize:
    def test_pca_arhmm_shapes_and_fit(self):
        model, true_params, z, x, inputs = switching_latents(jr.PRNGKey(5), num_sequences=2, num_timesteps=200)
        y = vmap(lambda k, xx, zz, u: vmap(lambda kk, xt, zt, ut: model.emission_component.sample_emission(kk, true_params.emissions, zt, xt, ut))(
            jr.split(k, 200), xx, zz, u))(jr.split(jr.PRNGKey(6), 2), x, z, inputs)
        params, props = model.initialize(jr.PRNGKey(7), method="pca_arhmm", emissions=y, num_init_iters=5)
        assert isinstance(params, ParamsSLDS)
        assert params.emissions.weights.shape == (1, 4, 2) and params.dynamics.weights.shape == (2, 2, 2)
        assert jnp.allclose(params.initial.probs.sum(), 1.0) and jnp.allclose(params.transitions.transition_matrix.sum(1), 1.0, atol=1e-5)
        fitted, lps = model.fit_em(params, props, y, num_iters=3, verbose=False)
        assert jnp.all(jnp.isfinite(lps)) and lps[-1] > lps[0]

    def test_pca_arhmm_needs_emissions(self):
        model = GaussianSLDS(2, 2, 4)
        with pytest.raises(ValueError):
            model.initialize(jr.PRNGKey(0), method="pca_arhmm")
        with pytest.raises(ValueError):
            model.initialize(jr.PRNGKey(0), method="nonsense")


def test_sticky_start():
    trans = StationarySLDSTransitions(3)
    params, _ = trans.initialize(jr.PRNGKey(0), method="sticky")
    P = params.transition_matrix
    assert jnp.allclose(P.sum(1), 1.0) and jnp.all(jnp.diag(P) > 0.9)
