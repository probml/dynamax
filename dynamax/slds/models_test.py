"""
Tests of ``GaussianSLDS``: sampling, fitting (monotone objective, damping, inputs, batches, per-state
emissions, covariance types), parameter freezing, jit, and the posterior summaries.
"""
import os
import subprocess
import sys

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap

from dynamax.parameters import ParameterProperties
from dynamax.slds import GaussianSLDS, ParamsSLDS, PosteriorSLDS
from dynamax.utils.utils import find_permutation, monotonically_increasing

X64 = jax.config.read("jax_enable_x64")


def well_separated_model(input_dim=0, single_subspace=True, **kwargs):
    """A K = 2 model whose states rotate in opposite directions around different points, so z is identifiable from y."""
    model = GaussianSLDS(2, 2, 4, input_dim, single_subspace=single_subspace, **kwargs)
    params, props = model.initialize(jr.PRNGKey(0))
    theta = 0.5
    rot = lambda s: 0.98 * jnp.array([[jnp.cos(s), -jnp.sin(s)], [jnp.sin(s), jnp.cos(s)]])
    params = params._replace(
        initial=params.initial._replace(probs=jnp.array([0.5, 0.5])),
        transitions=params.transitions._replace(transition_matrix=jnp.array([[0.95, 0.05], [0.05, 0.95]])),
        dynamics=params.dynamics._replace(weights=jnp.stack([rot(theta), rot(-theta)]),
                                          biases=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
                                          covs=jnp.tile(0.05 * jnp.eye(2), (2, 1, 1))),
        emissions=params.emissions._replace(covs=0.1 * jnp.ones_like(params.emissions.covs)))
    return model, params, props


class TestSampling:
    def test_shapes_and_states(self):
        model, params, _ = well_separated_model(input_dim=3)
        inputs = jr.normal(jr.PRNGKey(1), (50, 3))
        z, x, y = model.sample(params, jr.PRNGKey(2), 50, inputs)
        assert z.shape == (50,) and x.shape == (50, 2) and y.shape == (50, 4)
        assert jnp.all((z >= 0) & (z < 2))

    def test_sample_without_inputs(self):
        model, params, _ = well_separated_model()
        z, x, y = model.sample(params, jr.PRNGKey(2), 20)
        assert y.shape == (20, 4)


class TestPosterior:
    model, params, props = well_separated_model()
    z, x, y = model.sample(params, jr.PRNGKey(5), 200)

    def test_recovers_states_with_true_parameters(self):
        posterior, elbos = self.model.approximate_posterior(self.params, self.y, num_iters=10)
        posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
        z_hat = self.model.most_likely_states(self.params, self.y, posterior)
        accuracy = jnp.mean(z_hat == self.z)
        assert accuracy > 0.85
        assert jnp.mean(posterior.discrete.smoothed_probs.argmax(1) == self.z) > 0.85
        # x is identified up to the emission map, so compare the smoothed emissions
        assert jnp.mean((self.model.smooth(self.params, posterior) - self.y) ** 2) < jnp.mean((self.y - self.y.mean(0)) ** 2)

    def test_elbo_increases_over_sweeps(self):
        _, elbos = self.model.approximate_posterior(self.params, self.y, num_iters=10)
        assert monotonically_increasing(elbos, rtol=0 if X64 else 1e-5)

    def test_warm_start_continues(self):
        posterior, elbos = self.model.approximate_posterior(self.params, self.y, num_iters=4)
        posterior2, elbos2 = self.model.approximate_posterior(self.params, self.y, num_iters=4, posterior=posterior)
        assert elbos2[0] >= elbos[-1] - 1e-3 * jnp.abs(elbos[-1])

    def test_summaries(self):
        posterior, _ = self.model.approximate_posterior(self.params, self.y, num_iters=3)
        posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
        assert isinstance(posterior, PosteriorSLDS)
        assert self.model.smooth(self.params, posterior).shape == self.y.shape
        states, xs = self.model.sample_posterior(jr.PRNGKey(0), self.params, posterior, num_samples=5)
        assert states.shape == (5, 200) and xs.shape == (5, 200, 2)
        assert jnp.all(jnp.isfinite(xs))
        # Posterior samples of x scatter around the smoothed means with roughly the marginal variance
        states, xs = self.model.sample_posterior(jr.PRNGKey(1), self.params, posterior, num_samples=2000)
        assert jnp.allclose(xs.mean(0), posterior.continuous.smoothed_means, atol=0.1)
        assert jnp.allclose(xs.var(0), vmap(jnp.diag)(posterior.continuous.smoothed_covariances), rtol=0.25, atol=1e-3)
        assert jnp.allclose(jnp.mean(states == 1, axis=0), posterior.discrete.smoothed_probs[:, 1], atol=0.05)


def _sample_batch(key, num_sequences, num_timesteps):
    model, params, _ = well_separated_model()
    return vmap(lambda k: model.sample(params, k, num_timesteps)[2])(jr.split(key, num_sequences))


class TestFitEM:
    model, true_params, _ = well_separated_model()
    y = model.sample(true_params, jr.PRNGKey(6), 150)[2]
    y_batch = _sample_batch(jr.PRNGKey(7), 3, 60)

    def test_objective_increases_without_damping(self):
        model = GaussianSLDS(2, 2, 4, damping=0.0, perturb_unused_states=False)
        params, props = model.initialize(jr.PRNGKey(0))
        _, lps = model.fit_em(params, props, self.y, num_iters=15, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        assert monotonically_increasing(lps, rtol=0 if X64 else 1e-5)

    def test_fit_with_damping_and_batches(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        fitted, lps, posterior = model.fit_em(params, props, self.y_batch, num_iters=5, verbose=False, return_posterior=True)
        assert lps.shape == (5,) and jnp.all(jnp.isfinite(lps))
        assert lps[-1] > lps[0]
        assert posterior.discrete.smoothed_probs.shape == (3, 60, 2)
        assert posterior.continuous.smoothed_means.shape == (3, 60, 2)
        assert jnp.allclose(fitted.transitions.transition_matrix.sum(1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("dynamics_cov_type, emission_cov_type, single_subspace",
                             [("full", "diag", True), ("diag", "full", False), ("diag", "diag", False)])
    def test_variants_with_inputs(self, dynamics_cov_type, emission_cov_type, single_subspace):
        model = GaussianSLDS(2, 2, 4, input_dim=2, single_subspace=single_subspace,
                             dynamics_cov_type=dynamics_cov_type, emission_cov_type=emission_cov_type)
        params, props = model.initialize(jr.PRNGKey(1))
        inputs = jr.normal(jr.PRNGKey(2), (80, 2))
        _, _, y = model.sample(params, jr.PRNGKey(3), 80, inputs)
        params, props = model.initialize(jr.PRNGKey(4))
        fitted, lps = model.fit_em(params, props, y, inputs, num_iters=4, verbose=False)
        assert jnp.all(jnp.isfinite(lps)) and lps[-1] > lps[0]
        assert fitted.emissions.weights.shape == (1 if single_subspace else 2, 4, 2)
        assert fitted.dynamics.covs.shape == ((2, 2) if dynamics_cov_type == "diag" else (2, 2, 2))
        assert fitted.emissions.covs.shape[1:] == ((4,) if emission_cov_type == "diag" else (4, 4))
        if emission_cov_type == "diag":
            assert jnp.all(fitted.emissions.covs > 0)

    def test_verbose_matches_silent(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        _, lps_silent = model.fit_em(params, props, self.y, num_iters=4, verbose=False)
        _, lps_verbose = model.fit_em(params, props, self.y, num_iters=4, verbose=True, print_every=2)
        assert jnp.allclose(lps_silent, lps_verbose, rtol=1e-5)

    def test_zero_iterations(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        fitted, lps = model.fit_em(params, props, self.y, num_iters=0, verbose=False)
        assert lps.shape == (0,)
        assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(fitted), jax.tree_util.tree_leaves(params)))

    def test_damping_keeps_exact_zeros_finite(self):
        """Zero probabilities (left-to-right transitions, a one-hot initial distribution) survive damping."""
        model = GaussianSLDS(3, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        y = model.sample(params, jr.PRNGKey(1), 60)[2]
        params = params._replace(initial=params.initial._replace(probs=jnp.array([1.0, 0.0, 0.0])),
                                 transitions=params.transitions._replace(transition_matrix=jnp.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])))
        fitted, lps = model.fit_em(params, props, y, num_iters=3, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        assert jnp.array_equal(fitted.initial.probs, jnp.array([1.0, 0.0, 0.0]))
        assert jnp.all(fitted.transitions.transition_matrix[jnp.tril_indices(3, -1)] == 0.0)
        assert jnp.allclose(fitted.transitions.transition_matrix.sum(1), 1.0)

    def test_small_concentrations_stay_valid(self):
        """Dirichlet concentrations below one clip the pseudo-counts at zero instead of producing negative probabilities."""
        model = GaussianSLDS(3, 2, 3, transition_matrix_concentration=0.5, initial_probs_concentration=0.5, damping=0.0)
        params, props = model.initialize(jr.PRNGKey(0))
        y = model.sample(params, jr.PRNGKey(1), 60)[2]
        fitted, lps = model.fit_em(params, props, y, num_iters=3, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        assert jnp.all(fitted.transitions.transition_matrix >= 0) and jnp.all(fitted.initial.probs >= 0)

    def test_single_state_runs(self):
        model = GaussianSLDS(1, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        fitted, lps = model.fit_em(params, props, self.y, num_iters=3, verbose=False)
        assert jnp.all(jnp.isfinite(lps)) and jnp.isfinite(model.log_prior(fitted))
        assert fitted.transitions.transition_matrix.shape == (1, 1)


class TestFreezing:
    model, true_params, _ = well_separated_model()
    y = model.sample(true_params, jr.PRNGKey(8), 60)[2]

    @staticmethod
    def freeze(props, trainable):
        return jax.tree_util.tree_map(lambda p: ParameterProperties(trainable=trainable, constrainer=p.constrainer), props,
                                      is_leaf=lambda n: isinstance(n, ParameterProperties))

    def test_frozen_components_do_not_move(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        props = ParamsSLDS(initial=self.freeze(props.initial, False), transitions=self.freeze(props.transitions, False),
                           dynamics=props.dynamics, emissions=self.freeze(props.emissions, False))
        fitted, lps = model.fit_em(params, props, self.y, num_iters=3, verbose=False)
        assert jnp.array_equal(fitted.initial.probs, params.initial.probs)
        assert jnp.array_equal(fitted.transitions.transition_matrix, params.transitions.transition_matrix)
        assert all(jnp.array_equal(a, b) for a, b in zip(fitted.emissions, params.emissions))
        assert not jnp.array_equal(fitted.dynamics.weights, params.dynamics.weights)
        # the initial distribution of x is frozen by default
        assert jnp.array_equal(fitted.dynamics.initial_mean, params.dynamics.initial_mean)

    def test_trainable_initial_distribution_moves(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        props = props._replace(dynamics=props.dynamics._replace(initial_mean=ParameterProperties(),
                                                                initial_cov=ParameterProperties(constrainer=props.dynamics.initial_cov.constrainer)))
        fitted, _ = model.fit_em(params, props, self.y, num_iters=2, verbose=False)
        assert not jnp.array_equal(fitted.dynamics.initial_mean, params.dynamics.initial_mean)

    def test_partial_freezing_raises(self):
        model = GaussianSLDS(2, 2, 4)
        params, props = model.initialize(jr.PRNGKey(0))
        props = props._replace(dynamics=props.dynamics._replace(weights=ParameterProperties(trainable=False)))
        with pytest.raises(NotImplementedError):
            model.fit_em(params, props, self.y, num_iters=1, verbose=False)
        params, props = model.initialize(jr.PRNGKey(0))
        props = props._replace(emissions=props.emissions._replace(covs=ParameterProperties(trainable=False, constrainer=props.emissions.covs.constrainer)))
        with pytest.raises(NotImplementedError):
            model.fit_em(params, props, self.y, num_iters=1, verbose=False)


class TestIntractableMethods:
    model, params, _ = well_separated_model()
    y = model.sample(params, jr.PRNGKey(10), 10)[2]

    def test_exact_likelihoods_raise(self):
        with pytest.raises(NotImplementedError):
            self.model.marginal_log_prob(self.params, self.y)
        with pytest.raises(NotImplementedError):
            self.model.log_prob(self.params, jnp.zeros(10, int), self.y)
        with pytest.raises(NotImplementedError):
            self.model.fit_sgd(self.params, None, self.y)


def test_sample_posterior_transition_indexing():
    """A hand-set q(z) with transition matrices [I, fliplr(I)] gives z_0 == z_1 != z_2 in every sample, and q(x) draws
    come from the stored potentials."""
    from dynamax.slds import ContinuousPosteriorSLDS, DiscretePosteriorSLDS
    from dynamax.slds.inference import continuous_posterior_from_potentials
    from dynamax.linear_gaussian_ssm import BlockTridiagPotentials
    T, D = 3, 2
    model = GaussianSLDS(2, D, 4)
    params, _ = model.initialize(jr.PRNGKey(0))
    qz = DiscretePosteriorSLDS(initial_probs=jnp.array([0.5, 0.5]), log_transition_matrices=jnp.log(jnp.stack([jnp.eye(2), jnp.fliplr(jnp.eye(2))])),
                               log_likelihoods=jnp.zeros((T, 2)), log_normalizer=jnp.zeros(()), smoothed_probs=0.5 * jnp.ones((T, 2)),
                               trans_probs=jnp.stack([0.5 * jnp.eye(2), 0.5 * jnp.fliplr(jnp.eye(2))]))
    qx = continuous_posterior_from_potentials(BlockTridiagPotentials(jnp.tile(4.0 * jnp.eye(D), (T, 1, 1)), jnp.zeros((T - 1, D, D)), jnp.ones((T, D))))
    states, xs = model.sample_posterior(jr.PRNGKey(1), params, PosteriorSLDS(qz, qx), num_samples=2000)
    assert jnp.all(states[:, 0] == states[:, 1]) and jnp.all(states[:, 1] != states[:, 2])
    assert jnp.allclose(xs.mean(0), 0.25, atol=0.05) and jnp.allclose(xs.var(0), 0.25, atol=0.05)


class TestJit:
    model, true_params, _ = well_separated_model()
    y = model.sample(true_params, jr.PRNGKey(9), 40)[2]

    def test_e_step_jit_and_vmap(self):
        params, _ = self.model.initialize(jr.PRNGKey(0))
        stats, elbo = jit(self.model.e_step)(params, self.y)
        assert jnp.isfinite(elbo)
        ys = jnp.stack([self.y, self.y[::-1]])
        stats_b, elbos = vmap(self.model.e_step, in_axes=(None, 0))(params, ys)
        assert elbos.shape == (2,) and stats_b.dynamics.sum_w.shape == (2, 2)
        assert jnp.allclose(elbos[0], elbo, rtol=1e-4)

    def test_ssm_fit_em_cold_start_runs(self):
        """The generic SSM.fit_em (cold-start E-step every iteration) still works."""
        from dynamax.ssm import SSM
        params, props = self.model.initialize(jr.PRNGKey(0))
        fitted, lps = SSM.fit_em(self.model, params, props, self.y, num_iters=2, verbose=False)
        assert lps.shape == (2,) and jnp.all(jnp.isfinite(lps))


def recovery_problem(key=jr.PRNGKey(0), num_timesteps=2000):
    """K=3, D=2, N=10: states rotate by (k+1)·9° around distinct centers (with pure rotations the states are not
    identifiable even from the true parameters: accuracy 0.63)."""
    K, D, N = 3, 2, 10
    model = GaussianSLDS(K, D, N)
    params, _ = model.initialize(jr.PRNGKey(0))
    rot = lambda s: 0.95 * jnp.array([[jnp.cos(s), -jnp.sin(s)], [jnp.sin(s), jnp.cos(s)]])
    params = params._replace(
        transitions=params.transitions._replace(transition_matrix=0.9 * jnp.eye(K) + 0.1 / K),
        dynamics=params.dynamics._replace(weights=jnp.stack([rot((k + 1) * jnp.pi / 20) for k in range(K)]),
                                          biases=jnp.array([[1.0, 0.0], [-0.5, 0.8], [-0.5, -0.8]]),
                                          covs=jnp.tile(0.1 * jnp.eye(D), (K, 1, 1))),
        emissions=params.emissions._replace(covs=0.1 * jnp.ones((1, N))))
    z, x, y = model.sample(params, key, num_timesteps)
    return model, params, z, x, y


class TestRecovery:
    """Learning from the PCA + ARHMM initialization recovers a synthetic SLDS (invariants only: the latent space is
    identified up to an invertible linear map)."""
    model, true_params, z, x, y = recovery_problem()
    params, props = model.initialize(jr.PRNGKey(2), method="pca_arhmm", emissions=y)
    fitted, lps = model.fit_em(params, props, y, num_iters=60, verbose=False)
    posterior, _ = model.approximate_posterior(fitted, y, num_iters=5)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
    z_hat = model.most_likely_states(fitted, y, posterior)
    perm = find_permutation(z_hat, z)

    def test_discrete_states(self):
        assert jnp.mean(self.perm[self.z_hat] == self.z) > 0.85

    def test_dynamics_eigenvalues(self):
        for k in range(3):
            fitted_eigs = jnp.linalg.eigvals(self.fitted.dynamics.weights[k])
            true_eigs = jnp.linalg.eigvals(self.true_params.dynamics.weights[self.perm[k]])
            assert jnp.allclose(jnp.sort(jnp.abs(fitted_eigs)), jnp.sort(jnp.abs(true_eigs)), atol=0.05)
            assert jnp.allclose(jnp.sort(jnp.abs(jnp.angle(fitted_eigs))), jnp.sort(jnp.abs(jnp.angle(true_eigs))), atol=0.05)

    def test_emission_subspace_and_noise(self):
        cosines = jnp.linalg.svd(jnp.linalg.qr(self.fitted.emissions.weights[0])[0].T
                                 @ jnp.linalg.qr(self.true_params.emissions.weights[0])[0])[1]
        assert jnp.all(jnp.arccos(jnp.clip(cosines, -1, 1)) < 0.1)
        assert jnp.allclose(self.fitted.emissions.covs, self.true_params.emissions.covs, rtol=0.2)

    def test_latents(self):
        design = jnp.concatenate([self.posterior.continuous.smoothed_means, jnp.ones((self.x.shape[0], 1))], 1)
        coef = jnp.linalg.lstsq(design, self.x)[0]
        r2 = 1 - jnp.sum((design @ coef - self.x) ** 2) / jnp.sum((self.x - self.x.mean(0)) ** 2)
        assert r2 > 0.9

    def test_objective_increases(self):
        """With damping 0 and no unused-state perturbation every iteration is exact coordinate ascent."""
        model = GaussianSLDS(3, 2, 10, damping=0.0, perturb_unused_states=False)
        _, lps = model.fit_em(self.params, self.props, self.y, num_iters=20, verbose=False)
        assert monotonically_increasing(lps, rtol=0 if X64 else 1e-5)
        assert self.lps[-1] > self.lps[0]


def test_monotonicity_under_x64():
    """Re-run the monotonicity tests of this file with 64-bit floats, where they are strict."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "increases"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
test_monotonicity_under_x64.__test__ = not X64     # already under x64: nothing to re-run
