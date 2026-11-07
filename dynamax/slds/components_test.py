"""Tests of the SLDS components on their own: the Gaussian dynamics and emissions (log densities, expected
log densities under a Gaussian q(x), Hessian blocks, M-steps), the stationary transitions and the initial
distribution (priors, floored M-steps), and parameter validation. No model, no inference loop."""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from dynamax.linear_gaussian_ssm import LinearGaussianSSM, lgssm_to_block_tridiag
from dynamax.linear_gaussian_ssm.block_tridiag import BlockTridiagPotentials, block_tridiag_mvn_sample
from dynamax.slds import (
    LinearGaussianSLDSDynamics, LinearGaussianSLDSEmissions, SLDSInitialState, StationarySLDSTransitions)
from dynamax.slds.inference import continuous_posterior_from_potentials

X64 = jax.config.read("jax_enable_x64")
TOL = 1e-7 if X64 else 2e-3


def random_gaussian_posterior(key, num_timesteps, state_dim):
    """q(x) from random, diagonally dominant block-tridiagonal potentials."""
    k1, k2, k3 = jr.split(key, 3)
    T, D = num_timesteps, state_dim
    A = 0.3 * jr.normal(k1, (T, D, D))
    J_diag = 3.0 * jnp.eye(D) + A @ jnp.swapaxes(A, 1, 2)
    J_lower = 0.3 * jr.normal(k2, (T - 1, D, D))
    h = jr.normal(k3, (T, D))
    return continuous_posterior_from_potentials(BlockTridiagPotentials(J_diag, J_lower, h))


class TestGaussianPotentials:
    """The analytic Hessian blocks and the exact expectations of the Gaussian components (K=3, with inputs,
    single and per-state emission subspaces)."""
    num_states, state_dim, emission_dim, input_dim, num_timesteps = 3, 2, 5, 2, 20

    def _case(self, single_subspace):
        K, D, N, M, T = self.num_states, self.state_dim, self.emission_dim, self.input_dim, self.num_timesteps
        dyn = LinearGaussianSLDSDynamics(K, D, M, cov_type="full")
        emis = LinearGaussianSLDSEmissions(K, D, N, M, single_subspace=single_subspace, cov_type="full")
        dyn_params, _ = dyn.initialize(jr.PRNGKey(10))
        emis_params, _ = emis.initialize(jr.PRNGKey(11))
        inputs = jr.normal(jr.PRNGKey(12), (T, M))
        emissions = jr.normal(jr.PRNGKey(13), (T, N))
        Ez = jax.nn.softmax(jr.normal(jr.PRNGKey(14), (T, K)), axis=1)
        return dyn, emis, dyn_params, emis_params, inputs, emissions, Ez

    @pytest.mark.parametrize("single_subspace", [True, False])
    def test_hessian_blocks_match_autodiff(self, single_subspace):
        """(J_diag, J_lower) of the dynamics plus emissions equal the dense -Hessian of the expected log joint in x."""
        dyn, emis, dyn_params, emis_params, inputs, emissions, Ez = self._case(single_subspace)
        T, D = self.num_timesteps, self.state_dim

        def f(x):
            return dyn.expected_log_joint_x(dyn_params, x, Ez, inputs) + emis.expected_log_joint_x(emis_params, emissions, x, Ez, inputs)

        x = jr.normal(jr.PRNGKey(15), (T, D))
        dense = -jax.hessian(f)(x).reshape(T * D, T * D)
        J_diag, J_lower = dyn.neg_hessian_blocks(dyn_params, Ez, inputs)
        J_diag = J_diag + emis.neg_hessian_blocks(emis_params, emissions, x, Ez, inputs)
        for t in range(T):
            assert jnp.allclose(dense[t * D:(t + 1) * D, t * D:(t + 1) * D], J_diag[t], atol=100 * TOL, rtol=100 * TOL)
            if t < T - 1:
                assert jnp.allclose(dense[(t + 1) * D:(t + 2) * D, t * D:(t + 1) * D], J_lower[t], atol=100 * TOL, rtol=100 * TOL)
            if t < T - 2:
                assert jnp.allclose(dense[(t + 2) * D:(t + 3) * D, t * D:(t + 1) * D], 0.0, atol=100 * TOL)

    @pytest.mark.parametrize("single_subspace", [True, False])
    def test_expected_log_likelihoods(self, single_subspace):
        """The exact expectations reduce to the plug-in values at zero covariance and match Monte Carlo otherwise."""
        dyn, emis, dyn_params, emis_params, inputs, emissions, _ = self._case(single_subspace)
        T, D = self.num_timesteps, self.state_dim
        qx = random_gaussian_posterior(jr.PRNGKey(16), T, D)
        means = qx.smoothed_means
        # zero covariance: plug-in
        point = qx._replace(smoothed_covariances=jnp.zeros((T, D, D)),
                            smoothed_cross_covariances=jnp.einsum("ti,tj->tij", means[:-1], means[1:]))
        assert jnp.allclose(dyn.expected_log_likelihoods(dyn_params, point, inputs),
                            dyn.log_likelihoods(dyn_params, means, inputs), atol=100 * TOL, rtol=100 * TOL)
        assert jnp.allclose(emis.expected_log_likelihoods(emis_params, emissions, point, inputs),
                            emis.log_likelihoods(emis_params, emissions, means, inputs), atol=100 * TOL, rtol=100 * TOL)
        # Monte Carlo over q(x) with a fixed key: the summed error is within 4 standard errors
        potentials = BlockTridiagPotentials(qx.J_diag, qx.J_lower, qx.h)
        num_samples = 4000
        xs = vmap(lambda k: block_tridiag_mvn_sample(k, potentials))(jr.split(jr.PRNGKey(17), num_samples))
        for exact, per_sample in [
            (dyn.expected_log_likelihoods(dyn_params, qx, inputs), vmap(lambda x: dyn.log_likelihoods(dyn_params, x, inputs))(xs)),
            (emis.expected_log_likelihoods(emis_params, emissions, qx, inputs), vmap(lambda x: emis.log_likelihoods(emis_params, emissions, x, inputs))(xs))]:
            totals = per_sample.sum(axis=(1, 2))
            stderr = totals.std() / jnp.sqrt(num_samples)
            assert jnp.abs(exact.sum() - totals.mean()) < 4 * stderr + 1e-3 * jnp.abs(exact.sum())
            assert jnp.allclose(exact, per_sample.mean(0), atol=20 * per_sample.std(0).max() / jnp.sqrt(num_samples) + 1e-3)


class TestMStepsMatchLGSSM:
    """With one state and the priors switched off, the dynamics and emission M-steps on the exact posterior equal
    LinearGaussianSSM's maximum-likelihood M-step on the same sufficient statistics."""
    state_dim, emission_dim, input_dim, num_timesteps = 2, 4, 2, 40
    lgssm = LinearGaussianSSM(state_dim, emission_dim, input_dim, has_dynamics_bias=True, has_emissions_bias=True)
    lg_params, lg_props = lgssm.initialize(jr.PRNGKey(0))
    inputs = jr.normal(jr.PRNGKey(1), (num_timesteps, input_dim))
    _, emissions = lgssm.sample(lg_params, jr.PRNGKey(2), num_timesteps, inputs)
    dyn = LinearGaussianSLDSDynamics(1, state_dim, input_dim, cov_type="full",
                                     l2_penalty_weights=0.0, l2_penalty_input_weights=0.0, l2_penalty_bias=0.0, nu0=-(state_dim + 1), Psi0=0.0)
    emis = LinearGaussianSLDSEmissions(1, state_dim, emission_dim, input_dim, cov_type="full",
                                       l2_penalty=0.0, nu0=-(emission_dim + 1), Psi0=0.0)

    def test_m_steps(self):
        lg, T = self.lg_params, self.num_timesteps
        dyn_params, dyn_props = self.dyn.initialize(
            initial_mean=lg.initial.mean, initial_cov=lg.initial.cov, weights=lg.dynamics.weights[None],
            biases=lg.dynamics.bias[None], input_weights=lg.dynamics.input_weights[None], covs=lg.dynamics.cov[None])
        emis_params, emis_props = self.emis.initialize(
            weights=lg.emissions.weights[None], biases=lg.emissions.bias[None],
            input_weights=lg.emissions.input_weights[None], covs=lg.emissions.cov[None])
        # the exact posterior of the LGSSM, as a q(x) in information form
        potentials, _ = lgssm_to_block_tridiag(lg, self.emissions, self.inputs)
        qx = continuous_posterior_from_potentials(potentials)
        Ez = jnp.ones((T, 1))
        batch = lambda stats: jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], stats)
        new_dyn, _ = self.dyn.m_step(dyn_params, dyn_props, batch(self.dyn.collect_suff_stats(dyn_params, Ez, qx, self.inputs)), None)
        new_emis, _ = self.emis.m_step(emis_params, emis_props, batch(self.emis.collect_suff_stats(emis_params, self.emissions, Ez, qx, self.inputs)), None)
        lg_stats, _ = self.lgssm.e_step(lg, self.emissions, self.inputs)
        lg_new, _ = self.lgssm.m_step(lg, self.lg_props, batch(lg_stats), None)
        for ours, theirs in [(new_dyn.weights[0], lg_new.dynamics.weights), (new_dyn.biases[0], lg_new.dynamics.bias),
                             (new_dyn.input_weights[0], lg_new.dynamics.input_weights), (new_dyn.covs[0], lg_new.dynamics.cov),
                             (new_emis.weights[0], lg_new.emissions.weights), (new_emis.biases[0], lg_new.emissions.bias),
                             (new_emis.input_weights[0], lg_new.emissions.input_weights), (new_emis.covs[0], lg_new.emissions.cov)]:
            assert jnp.allclose(ours, theirs, atol=10 * TOL, rtol=10 * TOL)

    def test_log_prior_is_zero(self):
        """The maximum-likelihood hyperparameters switch the priors off."""
        dyn_params, _ = self.dyn.initialize(jr.PRNGKey(3))
        emis_params, _ = self.emis.initialize(jr.PRNGKey(4))
        assert jnp.allclose(self.dyn.log_prior(dyn_params), 0.0, atol=1e-6)
        assert jnp.allclose(self.emis.log_prior(emis_params), 0.0, atol=1e-6)


class TestDiscreteComponents:
    """ssm's floored normalized-count M-steps of the stationary transitions and the initial distribution."""

    def test_large_counts_keep_a_positive_floor(self):
        """The transition floor scales with the row total, so float32 does not flush it to zero."""
        trans = StationarySLDSTransitions(2)
        params, props = trans.initialize(jr.PRNGKey(0))
        new, _ = trans.m_step(params, props, jnp.array([[[3e6, 0.0], [10.0, 20.0]]]), None)
        assert jnp.all(new.transition_matrix > 0) and jnp.allclose(new.transition_matrix.sum(1), 1.0)

    def test_small_concentrations_clip_the_pseudo_counts(self):
        """Concentrations below one subtract from the counts; zero counts clip at zero instead of going negative."""
        trans = StationarySLDSTransitions(3, concentration=0.5)
        params, props = trans.initialize(jr.PRNGKey(0))
        new, _ = trans.m_step(params, props, jnp.array([[[5.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]]), None)
        assert jnp.all(new.transition_matrix >= 0) and jnp.allclose(new.transition_matrix.sum(1), 1.0)
        assert jnp.isfinite(trans.log_prior(new))
        initial = SLDSInitialState(3, initial_probs_concentration=0.5)
        params, props = initial.initialize(jr.PRNGKey(0))
        new, _ = initial.m_step(params, props, jnp.array([[4.0, 0.0, 0.0]]), None)
        assert jnp.all(new.probs >= 0) and jnp.allclose(new.probs.sum(), 1.0)

    def test_single_state_prior_is_zero(self):
        initial = SLDSInitialState(1)
        params, _ = initial.initialize(jr.PRNGKey(0))
        assert initial.log_prior(params) == 0.0


class TestValidationAndInversion:
    def test_wrong_covariance_shape_is_rejected(self):
        with pytest.raises(ValueError):
            LinearGaussianSLDSDynamics(2, 2).initialize(jr.PRNGKey(0), covs=0.1 * jnp.ones((2, 2)))
        with pytest.raises(ValueError):
            LinearGaussianSLDSEmissions(2, 2, 4).initialize(jr.PRNGKey(0), covs=jnp.tile(jnp.eye(4), (1, 1, 1)))

    def test_invert_recovers_the_latents(self):
        """Least-squares inversion of noiseless single-subspace emissions returns x."""
        emis = LinearGaussianSLDSEmissions(2, 2, 5, input_dim=1)
        params, _ = emis.initialize(jr.PRNGKey(0))
        x = jr.normal(jr.PRNGKey(1), (30, 2))
        inputs = jr.normal(jr.PRNGKey(2), (30, 1))
        y = x @ params.weights[0].T + inputs @ params.input_weights[0].T + params.biases[0]
        assert jnp.allclose(emis.invert(params, y, inputs), x, atol=100 * TOL)
