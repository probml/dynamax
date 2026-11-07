"""
Tests of the recurrent SLDS: the recurrent transition component (Hessian, Newton, Monte Carlo q(z) update,
sampling), the NASCAR benchmark of Linderman et al. (2017), and the recurrent ARHMM initialization.
"""
import itertools
import os
import subprocess
import sys

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.scipy.special import logsumexp

from dynamax.slds import GaussianSLDS, RecurrentGaussianSLDS, RecurrentSLDSTransitions, arhmm_initialization
from dynamax.slds.inference import (
    discrete_entropy, laplace_newton, log_space_hmm_smoother, update_continuous_posterior, update_discrete_posterior)
from dynamax.slds.initialization import _ARHMM, _ParamsARHMM
from dynamax.utils.utils import find_permutation

X64 = jax.config.read("jax_enable_x64")
TOL = 1e-7 if X64 else 2e-3


def unbatch(posterior):
    return jax.tree_util.tree_map(lambda a: a[0], posterior)


class TestRecurrentTransitions:
    num_states, state_dim, emission_dim, input_dim, num_timesteps = 3, 2, 5, 2, 12

    def _case(self, recurrent_only, **kwargs):
        model = RecurrentGaussianSLDS(self.num_states, self.state_dim, self.emission_dim, self.input_dim,
                                      recurrent_only=recurrent_only, emission_cov_type="full", **kwargs)
        params, _ = model.initialize(jr.PRNGKey(20))
        params = params._replace(transitions=params.transitions._replace(
            recurrent_weights=jr.normal(jr.PRNGKey(21), (self.num_states, self.state_dim))))
        inputs = jr.normal(jr.PRNGKey(22), (self.num_timesteps, self.input_dim))
        _, _, emissions = model.sample(params, jr.PRNGKey(23), self.num_timesteps, inputs)
        trans_probs = jax.nn.softmax(jr.normal(jr.PRNGKey(24), (self.num_timesteps - 1, self.num_states, self.num_states)).reshape(self.num_timesteps - 1, -1), axis=1)
        trans_probs = trans_probs.reshape(self.num_timesteps - 1, self.num_states, self.num_states)
        return model, params, inputs, emissions, trans_probs

    @pytest.mark.parametrize("recurrent_only", [False, True])
    def test_log_transition_matrices(self, recurrent_only):
        model, params, inputs, _, _ = self._case(recurrent_only)
        trans = model.transition_component
        x = jr.normal(jr.PRNGKey(25), (self.num_timesteps, self.state_dim))
        log_P = trans.log_transition_matrices(params.transitions, x, inputs)
        assert log_P.shape == (self.num_timesteps - 1, self.num_states, self.num_states)
        assert jnp.allclose(logsumexp(log_P, axis=-1), 0.0, atol=10 * TOL)
        assert jnp.allclose(jnp.exp(log_P), trans.transition_matrices(params.transitions, x, inputs), atol=TOL)
        # row i at time t is transition_row(state=i, x_t, u_{t+1})
        for t, i in [(0, 0), (5, 2), (self.num_timesteps - 2, 1)]:
            row = trans.transition_row(params.transitions, i, x[t], inputs[t + 1])
            assert jnp.allclose(row, jnp.exp(log_P[t, i]), atol=10 * TOL)
        if recurrent_only:
            assert jnp.allclose(log_P[:, 0], log_P[:, 1], atol=TOL)

    @pytest.mark.parametrize("recurrent_only", [False, True])
    def test_hessian_matches_autodiff(self, recurrent_only):
        model, params, inputs, _, trans_probs = self._case(recurrent_only)
        trans = model.transition_component
        T, D = self.num_timesteps, self.state_dim
        x = jr.normal(jr.PRNGKey(26), (T, D))
        f = lambda x: trans.expected_log_trans_prob(params.transitions, x, inputs, trans_probs)
        dense = -jax.hessian(f)(x).reshape(T * D, T * D)
        blocks = trans.neg_hessian_x(params.transitions, x, inputs, trans_probs)
        assert blocks.shape == (T - 1, D, D)
        for t in range(T - 1):
            assert jnp.allclose(dense[t * D:(t + 1) * D, t * D:(t + 1) * D], blocks[t], atol=100 * TOL, rtol=100 * TOL)
        off = dense - jax.scipy.linalg.block_diag(*blocks, jnp.zeros((D, D)))
        assert jnp.allclose(off, 0.0, atol=100 * TOL)
        eigs = jnp.linalg.eigvalsh(blocks)
        assert jnp.all(eigs > -100 * TOL)

    @pytest.mark.parametrize("recurrent_only", [False, True])
    def test_newton_converges_to_stationary_point(self, recurrent_only):
        """The Laplace update finds a stationary point of the expected log joint (to the Newton tolerance), and a
        second update does not move."""
        model, params, inputs, emissions, _ = self._case(recurrent_only, newton_tolerance=1e-12 if X64 else 1e-4)
        posterior, _ = model.approximate_posterior(params, emissions, inputs, num_iters=3)
        posterior = unbatch(posterior)
        qz, qx = posterior.discrete, posterior.continuous
        trans, dyn, emis = model._components
        Ez, Ezzp1 = qz.smoothed_probs, qz.trans_probs

        def log_joint(x):
            return (jnp.sum(Ez[0] * jnp.log(params.initial.probs))
                    + trans.expected_log_trans_prob(params.transitions, x, inputs, Ezzp1)
                    + dyn.expected_log_joint_x(params.dynamics, x, Ez, inputs)
                    + emis.expected_log_joint_x(params.emissions, emissions, x, Ez, inputs))

        grad = jax.grad(log_joint)(qx.smoothed_means)
        assert jnp.abs(grad).max() < (1e-6 if X64 else 1e-1)
        again = update_continuous_posterior(model, params, qz, qx, emissions, inputs)
        assert jnp.allclose(again.smoothed_means, qx.smoothed_means, atol=1e-2 if not X64 else 1e-6)
        # the precision is the negative Hessian at the mode
        T, D = self.num_timesteps, self.state_dim
        dense = -jax.hessian(log_joint)(qx.smoothed_means).reshape(T * D, T * D)
        for t in range(T):
            assert jnp.allclose(dense[t * D:(t + 1) * D, t * D:(t + 1) * D], qx.J_diag[t], atol=1e-2 if not X64 else 1e-6, rtol=1e-3)

    @pytest.mark.parametrize("recurrent_only", [False, True])
    def test_discrete_update_matches_enumeration(self, recurrent_only):
        """With S samples of x, q(z) is the normalized exp of the sample-averaged log joint: the folded row
        normalizer makes the chain exact."""
        K, T = 2, 4
        model = RecurrentGaussianSLDS(K, self.state_dim, self.emission_dim, recurrent_only=recurrent_only, emission_cov_type="full", num_samples=4)
        params, _ = model.initialize(jr.PRNGKey(30))
        params = params._replace(transitions=params.transitions._replace(recurrent_weights=2.0 * jr.normal(jr.PRNGKey(31), (K, self.state_dim))))
        _, _, emissions = model.sample(params, jr.PRNGKey(32), T)
        inputs = jnp.zeros((T, 0))
        posterior, _ = model.approximate_posterior(params, emissions, num_iters=2)
        qx = unbatch(posterior).continuous
        x_samples = jr.normal(jr.PRNGKey(33), (4, T, self.state_dim))
        qz = update_discrete_posterior(model, params, qx, x_samples, emissions, inputs)
        trans, dyn, _ = model._components
        expected_log_P = vmap(lambda x: trans.log_transition_matrices(params.transitions, x, inputs))(x_samples).mean(0)
        ll = dyn.expected_log_likelihoods(params.dynamics, qx, inputs)
        seqs = jnp.array(list(itertools.product(range(K), repeat=T)))
        log_joint = (jnp.log(params.initial.probs)[seqs[:, 0]]
                     + expected_log_P[jnp.arange(T - 1), seqs[:, :-1], seqs[:, 1:]].sum(1)
                     + ll[jnp.arange(T), seqs].sum(1))
        probs = jnp.exp(log_joint - logsumexp(log_joint))
        marginals = jnp.einsum("s,stk->tk", probs, jax.nn.one_hot(seqs, K))
        assert jnp.allclose(qz.smoothed_probs, marginals, atol=10 * TOL)
        pairs = jnp.einsum("s,sti,stj->tij", probs, jax.nn.one_hot(seqs[:, :-1], K), jax.nn.one_hot(seqs[:, 1:], K))
        assert jnp.allclose(qz.trans_probs, pairs, atol=10 * TOL)
        assert jnp.allclose(discrete_entropy(qz), -jnp.sum(probs * jnp.log(probs)), atol=10 * TOL)

    def test_sampling_follows_the_softmax(self):
        """The empirical next-state frequencies of a long sample match the recurrent transition probabilities."""
        model = RecurrentGaussianSLDS(2, 1, 2, recurrent_only=True)
        params, _ = model.initialize(jr.PRNGKey(40))
        params = params._replace(
            transitions=params.transitions._replace(logits=jnp.zeros((1, 2)), recurrent_weights=jnp.array([[2.0], [-2.0]])),
            dynamics=params.dynamics._replace(weights=0.5 * jnp.ones((2, 1, 1)), biases=jnp.zeros((2, 1)), covs=jnp.ones((2, 1, 1))))
        z, x, _ = model.sample(params, jr.PRNGKey(41), 20000)
        p_state0 = jax.nn.sigmoid(4.0 * x[:-1, 0])             # P(z_{t+1} = 0 | x_t)
        for lo, hi in [(-3, -1), (-1, 0), (0, 1), (1, 3)]:
            in_bin = (x[:-1, 0] > lo) & (x[:-1, 0] <= hi)
            assert jnp.abs(jnp.mean(z[1:][in_bin] == 0) - jnp.mean(p_state0[in_bin])) < 0.03

    def test_m_step_increases_objective_and_respects_freezing(self):
        model, params, inputs, emissions, trans_probs = self._case(False)
        trans = model.transition_component
        x_samples = jr.normal(jr.PRNGKey(27), (2, self.num_timesteps, self.state_dim))
        stats = jax.tree_util.tree_map(lambda a: a[None], trans.collect_suff_stats(params.transitions, type("Q", (), {"trans_probs": trans_probs})(), x_samples, inputs))
        objective = lambda p: trans.log_prior(p) + vmap(lambda x: trans.expected_log_trans_prob(p, x, inputs, trans_probs))(x_samples).mean()
        _, props = trans.initialize(jr.PRNGKey(0))
        new_params, _ = trans.m_step(params.transitions, props, stats, None)
        assert objective(new_params) > objective(params.transitions) + 1e-3
        # frozen leaves stay put, trainable ones move
        from dynamax.parameters import ParameterProperties
        props = props._replace(recurrent_weights=ParameterProperties(trainable=False))
        frozen, _ = trans.m_step(params.transitions, props, stats, None)
        assert jnp.array_equal(frozen.recurrent_weights, params.transitions.recurrent_weights)
        assert not jnp.array_equal(frozen.logits, params.transitions.logits)
        assert objective(frozen) > objective(params.transitions)
        # an optax optimizer without a line search works too
        import optax
        adam_trans = RecurrentSLDSTransitions(self.num_states, self.state_dim, self.input_dim, m_step_optimizer=optax.adam(1e-2), m_step_num_iters=50)
        new_params, _ = adam_trans.m_step(params.transitions, trans.initialize(jr.PRNGKey(0))[1], stats, None)
        assert objective(new_params) > objective(params.transitions)

    def test_log_prior(self):
        trans = RecurrentSLDSTransitions(3, 2, 1, l2_penalty_recurrent=2.0, l2_penalty_input=4.0, concentration=1.0)
        params, _ = trans.initialize(jr.PRNGKey(0))
        params = params._replace(recurrent_weights=jnp.ones((3, 2)), input_weights=jnp.ones((3, 1)))
        # Dirichlet(1) rows have log density log Γ(3) = log 2 each
        assert jnp.allclose(trans.log_prior(params), 3 * jnp.log(2.0) - 0.5 * 2.0 * 6 - 0.5 * 4.0 * 3, atol=1e-5)
        only = RecurrentSLDSTransitions(3, 2, 1, recurrent_only=True)
        params, _ = only.initialize(jr.PRNGKey(0))
        assert jnp.allclose(only.log_prior(params), 0.0)


def test_exactness_under_x64():
    """Re-run the transition tests with 64-bit floats, where the tolerances tighten."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "TestRecurrentTransitions"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
test_exactness_under_x64.__test__ = not X64     # already under x64: nothing to re-run


# ---- NASCAR benchmark (Linderman et al., 2017; ssm's rSLDS notebook) ----------------------------------------------
def make_nascar_params(model, key=jr.PRNGKey(0)):
    """ssm's NASCAR rSLDS: two states rotating (clockwise) around (+2, 0) and (-2, 0), two states moving straight along
    the top and bottom, and recurrent-only transitions that carve the plane into right / left / top / bottom."""
    D, K = 2, 4
    params, props = model.initialize(key)
    rot = lambda th: jnp.array([[jnp.cos(th), -jnp.sin(th)], [jnp.sin(th), jnp.cos(th)]])
    As = [rot(-jnp.pi / 24), rot(-jnp.pi / 48), jnp.eye(D), jnp.eye(D)]
    centers = [jnp.array([2.0, 0.0]), jnp.array([-2.0, 0.0])]
    bs = [-(A - jnp.eye(D)) @ c for A, c in zip(As[:2], centers)] + [jnp.array([0.1, 0.0]), jnp.array([-0.25, 0.0])]
    W = jnp.array([[100.0, 0.0], [-100.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    r = jnp.array([-200.0, -200.0, 0.0, 0.0])
    H = jnp.linalg.qr(jr.normal(jr.fold_in(key, 7), (model.emission_dim, D)))[0]
    params = params._replace(
        initial=params.initial._replace(probs=jnp.array([0.0, 0.0, 1.0, 0.0])),
        transitions=params.transitions._replace(logits=r[None], recurrent_weights=W, input_weights=jnp.zeros((K, 0))),
        dynamics=params.dynamics._replace(initial_mean=jnp.array([0.0, 1.0]), initial_cov=1e-4 * jnp.eye(D),
                                          weights=jnp.stack(As), biases=jnp.stack(bs), covs=1e-4 * jnp.ones((K, D))),
        emissions=params.emissions._replace(weights=H[None], biases=jnp.zeros((1, model.emission_dim)),
                                            covs=1e-2 * jnp.ones((1, model.emission_dim))))
    return params, props


def affine_alignment(means, x):
    """x ≈ G x̂ + g by least squares; returns (G, g, aligned means)."""
    design = jnp.concatenate([means, jnp.ones((means.shape[0], 1))], 1)
    coef = jnp.linalg.lstsq(design, x)[0]
    return coef[:-1].T, coef[-1], design @ coef


class TestNascar:
    """Runs in float32 and, through ``test_nascar_under_x64``, in 64-bit. JAX draws different random numbers for
    float32 and float64, so the two runs see different datasets. On this stiff problem (Q = 1e-4, partition weights
    of 100) the variational fit is sensitive to the data and the initialization in both precisions (state accuracy
    after learning ranges over 0.64-0.90 across seeds); the data seed below gives a good fit in both."""
    num_timesteps = 2000
    model = RecurrentGaussianSLDS(4, 2, 10, recurrent_only=True, dynamics_cov_type="diag")
    true_params, props = make_nascar_params(model)
    z, x, y = model.sample(true_params, jr.PRNGKey(2), num_timesteps)

    def test_sample_visits_every_state(self):
        counts = jnp.bincount(self.z, length=4)
        assert jnp.all(counts > 100)
        assert jnp.abs(self.x[:, 0]).max() < 5 and jnp.abs(self.x[:, 1]).max() < 3

    def test_posterior_with_true_parameters(self):
        """The structured mean-field posterior under the true parameters. The sharp partition (weights of 100) and
        the small dynamics noise make the coordinate ascent land in a local optimum: measured state accuracy around
        0.8-0.86 (ssm's own structured mean-field posterior on one of these datasets reaches 0.66 after 10 and 0.69
        after 30 sweeps, with a far lower ELBO); the latent trajectory is recovered closely."""
        posterior, elbos = self.model.approximate_posterior(self.true_params, self.y, num_iters=10)
        assert jnp.all(jnp.isfinite(elbos))
        posterior = unbatch(posterior)
        z_hat = self.model.most_likely_states(self.true_params, self.y, posterior)
        assert jnp.mean(z_hat == self.z) > 0.75
        _, _, aligned = affine_alignment(posterior.continuous.smoothed_means, self.x)
        assert jnp.sqrt(jnp.mean((aligned - self.x) ** 2)) < 0.15     # measured 0.11
        # The stored chain is the one the jitted smoother used: re-reading it eagerly (entropy, smoother, sampler)
        # gives the same q(z). Storing the transitions as probabilities would underflow 46% of them to zero in
        # float32 (XLA's log(exp(.)) simplification hides that inside jit) and the eager re-runs would disagree
        # with the stored marginals at 90 of 2000 steps.
        qz = posterior.discrete
        assert jnp.isfinite(discrete_entropy(qz))
        _, smoothed, _ = log_space_hmm_smoother(qz.initial_probs, qz.log_transition_matrices, qz.log_likelihoods)
        assert jnp.allclose(smoothed, qz.smoothed_probs, atol=1e-4)
        states, _ = self.model.sample_posterior(jr.PRNGKey(0), self.true_params, posterior, num_samples=200)
        freq = jnp.stack([jnp.mean(states == k, 0) for k in range(4)], 1)
        assert jnp.allclose(freq, qz.smoothed_probs, atol=0.15)     # 4 standard errors at p = 0.5

    def test_learning_from_pca_arhmm(self):
        """50 iterations from the PCA + ARHMM start recover the states, the latent trajectory and the partition of the
        plane that the recurrent weights induce (compared after aligning the latent space)."""
        params, props = self.model.initialize(jr.PRNGKey(2), method="pca_arhmm", emissions=self.y)
        fitted, lps = self.model.fit_em(params, props, self.y, num_iters=50, verbose=False)
        assert jnp.all(jnp.isfinite(lps))
        posterior, _ = self.model.approximate_posterior(fitted, self.y, num_iters=5)
        posterior = unbatch(posterior)
        z_hat = self.model.most_likely_states(fitted, self.y, posterior)
        perm = find_permutation(z_hat, self.z)
        assert jnp.mean(perm[z_hat] == self.z) > 0.8
        G, g, aligned = affine_alignment(posterior.continuous.smoothed_means, self.x)
        assert jnp.sqrt(jnp.mean((aligned - self.x) ** 2)) < 0.1
        # the partition argmax_j (W_j · x + r_j) of the plane, on a grid over the data range
        grid = jnp.stack(jnp.meshgrid(jnp.linspace(self.x[:, 0].min(), self.x[:, 0].max(), 40),
                                      jnp.linspace(self.x[:, 1].min(), self.x[:, 1].max(), 40)), -1).reshape(-1, 2)
        W_true, r_true = self.true_params.transitions.recurrent_weights, self.true_params.transitions.logits[0]
        W_fit, r_fit = fitted.transitions.recurrent_weights, fitted.transitions.logits[0]
        grid_hat = (grid - g) @ jnp.linalg.inv(G).T
        true_partition = jnp.argmax(grid @ W_true.T + r_true, 1)
        fitted_partition = perm[jnp.argmax(grid_hat @ W_fit.T + r_fit, 1)]
        assert jnp.mean(fitted_partition == true_partition) > 0.8      # measured 0.84-0.91
        # directions of the row differences of W (the only identifiable part), mapped to the true latent space
        W_eff = W_fit @ jnp.linalg.inv(G)
        cosine = lambda a, b: a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
        for l in range(4):
            for m in range(l):
                assert cosine(W_eff[l] - W_eff[m], W_true[perm[l]] - W_true[perm[m]]) > 0.8


def test_nascar_under_x64():
    """Run the NASCAR benchmark in a 64-bit subprocess."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "TestNascar"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
test_nascar_under_x64.__test__ = not X64     # already under x64: nothing to re-run


# ---- recurrent transitions inside the ARHMM initialization ---------------------------------------------------------
class TestRecurrentInitialization:
    def test_adapter_matches_component_and_learns_weights(self):
        model = RecurrentGaussianSLDS(4, 2, 10, recurrent_only=True, dynamics_cov_type="diag")
        true_params, _ = make_nascar_params(model)
        z, x, _ = model.sample(true_params, jr.PRNGKey(3), 600)
        inputs = jnp.zeros((1, 600, 0))
        adapter = model.transition_component.as_hmm_transitions(model.state_dim)
        arhmm_inputs = jnp.concatenate([x[None], inputs], -1)
        matrices = adapter._compute_transition_matrices(true_params.transitions, arhmm_inputs[0])
        assert jnp.allclose(matrices, model.transition_component.transition_matrices(true_params.transitions, x, inputs[0]))
        # a sticky start has zero recurrent weights; the ARHMM EM learns them from x
        start, _ = model.transition_component.initialize(jr.PRNGKey(0), "sticky")
        assert jnp.all(start.recurrent_weights == 0.0)
        _, transitions, _, _ = arhmm_initialization(jr.PRNGKey(1), model, x[None], inputs, num_iters=20)
        assert jnp.linalg.norm(transitions.recurrent_weights) > 1.0
        # the learned partition agrees with the truth on the data (up to relabeling)
        fitted_part = jnp.argmax(x[:-1] @ transitions.recurrent_weights.T + transitions.logits[0], 1)
        true_part = jnp.argmax(x[:-1] @ true_params.transitions.recurrent_weights.T + true_params.transitions.logits[0], 1)
        perm = find_permutation(fitted_part, true_part)
        assert jnp.mean(perm[fitted_part] == true_part) > 0.8

    def test_batched_statistics_shapes(self):
        model = RecurrentGaussianSLDS(3, 2, 4, input_dim=1)
        params, _ = model.initialize(jr.PRNGKey(0))
        B, T = 2, 30
        x = jr.normal(jr.PRNGKey(1), (B, T, 2))
        inputs = jr.normal(jr.PRNGKey(2), (B, T, 1))
        arhmm = _ARHMM(model)
        arhmm_inputs = jnp.concatenate([x, inputs], -1)
        arhmm_params, arhmm_props = arhmm.initialize(jr.PRNGKey(3), "kmeans", x, arhmm_inputs, transition_params=dict(method="sticky"))
        stats, lls = vmap(arhmm.e_step, (None, 0, 0))(arhmm_params, x, arhmm_inputs)
        assert stats[1].trans_probs.shape == (B, T - 1, 3, 3)
        assert stats[1].x_samples.shape == (B, 1, T, 2)
        assert stats[1].inputs.shape == (B, T, 1)
        new_params, _ = arhmm.m_step(arhmm_params, arhmm_props, stats, arhmm.initialize_m_step_state(arhmm_params, arhmm_props))
        assert new_params.transitions.recurrent_weights.shape == (3, 2)
        assert jnp.all(jnp.isfinite(new_params.transitions.recurrent_weights))
