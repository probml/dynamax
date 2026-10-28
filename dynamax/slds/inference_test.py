"""
Tests for the Laplace-EM posterior updates of the SLDS: exactness with one discrete state against the
Kalman smoother, the ELBO against a brute-force marginal likelihood, and the Newton line search.
``test_exactness_under_x64`` re-runs this file in a 64-bit subprocess with tight tolerances.
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

from dynamax.hidden_markov_model.inference import hmm_filter, hmm_smoother
from dynamax.parameters import ParameterProperties
from dynamax.linear_gaussian_ssm import LinearGaussianSSM, lgssm_filter
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from dynamax.slds import GaussianSLDS
from dynamax.slds.inference import (
    backtracking_line_search, initialize_posterior, log_space_hmm_posterior_sample, log_space_hmm_smoother, slds_elbo)

X64 = jax.config.read("jax_enable_x64")
TOL = 1e-7 if X64 else 2e-3


def mle_kwargs(state_dim, emission_dim):
    """Hyperparameters that turn every prior off (so that the M-step is maximum likelihood)."""
    return dict(dynamics_kwargs=dict(l2_penalty_weights=0.0, l2_penalty_input_weights=0.0, l2_penalty_biases=0.0,
                                     nu0=-(state_dim + 1), Psi0=0.0),
                emission_kwargs=dict(l2_penalty=0.0, nu0=-(emission_dim + 1), Psi0=0.0))


def to_lgssm(params, state=0):
    """The LGSSM parameters of one discrete state of a single-subspace SLDS."""
    dyn, emis = params.dynamics, params.emissions
    return ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=dyn.initial_mean, cov=dyn.initial_cov),
        dynamics=ParamsLGSSMDynamics(weights=dyn.weights[state], bias=dyn.biases[state],
                                     input_weights=dyn.input_weights[state], cov=dyn.covs[state]),
        emissions=ParamsLGSSMEmissions(weights=emis.weights[0], bias=emis.biases[0],
                                       input_weights=emis.input_weights[0], cov=emis.covs[0]))


def unbatch(posterior):
    """The posterior of the first (only) sequence in a batched posterior."""
    return jax.tree_util.tree_map(lambda a: a[0], posterior)


class TestSingleState:
    """With K = 1 the variational posterior is the exact LGSSM posterior and the ELBO is log p(y)."""
    state_dim, emission_dim, input_dim, num_timesteps = 2, 4, 2, 40
    model = GaussianSLDS(1, state_dim, emission_dim, input_dim, emission_cov_type="full", damping=0.0,
                         **mle_kwargs(state_dim, emission_dim))
    params, props = model.initialize(jr.PRNGKey(0))
    inputs = jr.normal(jr.PRNGKey(1), (num_timesteps, input_dim))
    _, _, emissions = model.sample(params, jr.PRNGKey(2), num_timesteps, inputs)
    lgssm = LinearGaussianSSM(state_dim, emission_dim, input_dim, has_dynamics_bias=True, has_emissions_bias=True)
    reference = lgssm.smoother(to_lgssm(params), emissions, inputs)

    def test_posterior_matches_kalman_smoother(self):
        """Means, covariances and cross-covariances of q(x) equal the smoother's after one sweep."""
        posterior, _ = self.model.approximate_posterior(self.params, self.emissions, self.inputs, num_iters=1)
        qx = unbatch(posterior).continuous
        assert jnp.allclose(qx.smoothed_means, self.reference.smoothed_means, atol=TOL)
        assert jnp.allclose(qx.smoothed_covariances, self.reference.smoothed_covariances, atol=TOL)
        assert jnp.allclose(qx.smoothed_cross_covariances, self.reference.smoothed_cross_covariances, atol=TOL)

    def test_elbo_is_marginal_log_likelihood(self):
        """The ELBO is tight: it equals the LGSSM marginal log likelihood."""
        _, elbos = self.model.approximate_posterior(self.params, self.emissions, self.inputs, num_iters=2)
        assert jnp.allclose(elbos[-1], self.reference.marginal_loglik, rtol=TOL)

    def test_em_step_matches_lgssm(self):
        """One maximum-likelihood M-step equals LinearGaussianSSM's on the dynamics and emissions (the SLDS keeps
        the initial distribution of x frozen; the LGSSM M-step cannot freeze it alone)."""
        fitted, lps = self.model.fit_em(self.params, self.props, self.emissions, self.inputs, num_iters=1, verbose=False)
        _, lg_props = self.lgssm.initialize(jr.PRNGKey(0))
        lg_fitted, lg_lps = self.lgssm.fit_em(to_lgssm(self.params), lg_props, self.emissions, self.inputs, num_iters=1, verbose=False)
        assert jnp.allclose(lps, lg_lps, rtol=TOL)
        ours, theirs = to_lgssm(fitted), lg_fitted
        for a, b in zip(jax.tree_util.tree_leaves((ours.dynamics, ours.emissions)), jax.tree_util.tree_leaves((theirs.dynamics, theirs.emissions))):
            assert jnp.allclose(a, b, atol=10 * TOL, rtol=10 * TOL)


    def test_em_trace_matches_lgssm(self):
        """With the initial distribution of x trainable too, five EM iterations reproduce LinearGaussianSSM's log-probability trace."""
        props = self.props._replace(dynamics=self.props.dynamics._replace(
            initial_mean=ParameterProperties(), initial_cov=ParameterProperties(constrainer=self.props.dynamics.initial_cov.constrainer)))
        _, lps = self.model.fit_em(self.params, props, self.emissions, self.inputs, num_iters=5, verbose=False)
        _, lg_props = self.lgssm.initialize(jr.PRNGKey(0))
        _, lg_lps = self.lgssm.fit_em(to_lgssm(self.params), lg_props, self.emissions, self.inputs, num_iters=5, verbose=False)
        assert jnp.allclose(lps, lg_lps, rtol=10 * TOL)

    def test_stiff_dynamics_in_float32(self):
        """Nearly deterministic dynamics (Q = 1e-4, T = 500): the information-form smoother stays within 1e-2 of the Kalman smoother."""
        D, N, T = 2, 4, 500
        model = GaussianSLDS(1, D, N, emission_cov_type="full", damping=0.0)
        params, _ = model.initialize(jr.PRNGKey(0))
        params = params._replace(dynamics=params.dynamics._replace(covs=1e-4 * jnp.tile(jnp.eye(D), (1, 1, 1)),
                                                                   weights=0.999 * jnp.tile(jnp.eye(D), (1, 1, 1))))
        _, _, emissions = model.sample(params, jr.PRNGKey(3), T)
        posterior, elbos = model.approximate_posterior(params, emissions, num_iters=1)
        reference = LinearGaussianSSM(D, N, has_dynamics_bias=True, has_emissions_bias=True).smoother(to_lgssm(params), emissions)
        means = posterior.continuous.smoothed_means[0]
        assert jnp.all(jnp.isfinite(means))
        assert jnp.abs(means - reference.smoothed_means).max() < 1e-2 * jnp.abs(reference.smoothed_means).max()
        if X64:
            assert jnp.allclose(elbos[-1], reference.marginal_loglik, rtol=1e-4)


class TestTwoStates:
    """A K = 2 model small enough to enumerate the discrete sequences."""
    num_states, state_dim, emission_dim, num_timesteps = 2, 2, 3, 5
    model = GaussianSLDS(num_states, state_dim, emission_dim, emission_cov_type="full", damping=0.0)
    params, props = model.initialize(jr.PRNGKey(3))
    params = params._replace(transitions=params.transitions._replace(transition_matrix=jnp.array([[0.9, 0.1], [0.2, 0.8]])))
    _, _, emissions = model.sample(params, jr.PRNGKey(4), num_timesteps)
    inputs = jnp.zeros((num_timesteps, 0))

    def _log_marginal_likelihood(self):
        """log p(y) by summing the LGSSM likelihood of every discrete sequence (time-varying dynamics)."""
        dyn, emis, P = self.params.dynamics, self.params.emissions, self.params.transitions.transition_matrix
        log_P, log_pi = jnp.log(P), jnp.log(self.params.initial.probs)

        def _one(z):
            z = jnp.array(z)
            tv = ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=dyn.initial_mean, cov=dyn.initial_cov),
                dynamics=ParamsLGSSMDynamics(weights=dyn.weights[z], bias=dyn.biases[z], input_weights=dyn.input_weights[z], cov=dyn.covs[z]),
                emissions=ParamsLGSSMEmissions(weights=emis.weights[0], bias=emis.biases[0], input_weights=emis.input_weights[0], cov=emis.covs[0]))
            log_pz = log_pi[z[0]] + log_P[z[:-1], z[1:]].sum()
            return log_pz + lgssm_filter(tv, self.emissions, self.inputs).marginal_loglik

        return logsumexp(jnp.array([_one(z) for z in itertools.product(range(self.num_states), repeat=self.num_timesteps)]))

    def test_elbo_is_a_lower_bound(self):
        """The converged ELBO is below log p(y) and increases over the sweeps."""
        _, elbos = self.model.approximate_posterior(self.params, self.emissions, num_iters=20)
        assert jnp.all(jnp.isfinite(elbos))
        assert elbos[-1] <= self._log_marginal_likelihood() + 1e-3
        assert jnp.all(jnp.diff(elbos) >= -1e-3 * jnp.abs(elbos[:-1]))


class TestNewton:
    """The backtracking line search of the Newton solver."""
    num_timesteps, dim = 6, 2
    x_star = jr.normal(jr.PRNGKey(1), (num_timesteps, dim))

    def test_line_search_backtracks(self):
        """A direction that overshoots is shortened; NaN objective values are treated as no decrease."""
        f = lambda x: jnp.where(jnp.all(jnp.abs(x) < 10.0), jnp.sum(x ** 2), jnp.nan)
        x = jnp.ones_like(self.x_star)
        dx = -20.0 * x   # the full step lands at -19 x, where f is NaN
        g = jax.grad(lambda x: jnp.sum(x ** 2))(x)
        step = backtracking_line_search(f, x, dx, g)
        assert 0.0 < step < 1.0
        assert f(x + step * dx) <= f(x) + 0.2 * step * jnp.sum(g * dx)


class TestLogSpaceChain:
    """The log-space smoother and sampler of q(z) against enumeration, on a chain that kills probability-space messages."""

    def test_sampler_matches_enumeration(self):
        """The eight path frequencies of a 3-step, 2-state chain match the exact joint, as do the smoother's outputs."""
        pi0 = jnp.array([0.7, 0.3])
        P = jnp.array([[[0.9, 0.1], [0.4, 0.6]], [[0.2, 0.8], [0.5, 0.5]]])
        ll = jnp.log(jnp.array([[0.5, 0.2], [0.1, 0.9], [0.6, 0.3]]))
        paths = jnp.array(list(itertools.product(range(2), repeat=3)))
        log_joint = (jnp.log(pi0)[paths[:, 0]] + jnp.log(P[0])[paths[:, 0], paths[:, 1]] + jnp.log(P[1])[paths[:, 1], paths[:, 2]]
                     + ll[0, paths[:, 0]] + ll[1, paths[:, 1]] + ll[2, paths[:, 2]])
        exact = jax.nn.softmax(log_joint)
        num_samples = 20000
        samples = vmap(lambda k: log_space_hmm_posterior_sample(k, pi0, jnp.log(P), ll))(jr.split(jr.PRNGKey(0), num_samples))
        freq = jnp.bincount(samples @ jnp.array([4, 2, 1]), length=8) / num_samples
        assert jnp.all(jnp.abs(freq - exact) < 5 * jnp.sqrt(exact * (1 - exact) / num_samples))   # per path, 5 standard errors
        log_Z, smoothed, trans = log_space_hmm_smoother(pi0, jnp.log(P), ll)
        assert jnp.allclose(log_Z, logsumexp(log_joint), rtol=1e-5)
        assert jnp.allclose(smoothed[:, 1], jnp.stack([exact[paths[:, t] == 1].sum() for t in range(3)]), atol=1e-5)
        assert jnp.allclose(trans[0, 0, 1], exact[(paths[:, 0] == 0) & (paths[:, 1] == 1)].sum(), atol=1e-5)

    def test_sampler_survives_underflowing_messages(self):
        """A left-to-right chain 0 -> 1 -> 2 whose evidence (gaps of 150 nats) favors state 0 and then state 2:
        the posterior must pass through state 1 for exactly one step around the switch. In float32 the
        probability-space filter underflows state 1 to exactly zero while the evidence favors 0, so at the
        switch it has no mass to move to 2 and the row dies. The log-space smoother and sampler match the
        exact enumeration over the two switch times."""
        T, K, gap = 40, 3, 150.0
        pi0 = jnp.array([1.0, 0.0, 0.0])
        P = jnp.array([[1 - 1e-3, 1e-3, 0.0], [0.0, 1 - 1e-3, 1e-3], [0.0, 0.0, 1.0]])
        ll = gap * jax.nn.one_hot(jnp.where(jnp.arange(T) < 20, 0, 2), K)
        if not X64:
            assert jnp.any(hmm_filter(pi0, P, ll).filtered_probs.sum(1) == 0)
        # every monotone path: 0 before t1, 1 on [t1, t2), 2 from t2 on, with 1 <= t1 <= t2 <= T
        t1, t2 = jnp.array([(a, b) for a in range(1, T + 1) for b in range(a, T + 1)]).T
        paths = jnp.where(jnp.arange(T) < t1[:, None], 0, jnp.where(jnp.arange(T) < t2[:, None], 1, 2))
        log_joint = jnp.log(P)[paths[:, :-1], paths[:, 1:]].sum(1) + ll[jnp.arange(T), paths].sum(1)
        exact = jax.nn.softmax(log_joint)
        exact_marginals = jnp.stack([(exact[:, None] * (paths == k)).sum(0) for k in range(K)], 1)
        _, smoothed, _ = log_space_hmm_smoother(pi0, jnp.log(P), ll)
        assert jnp.allclose(smoothed, exact_marginals, atol=1e-3)
        assert 0.4 < smoothed[19, 1] < 0.6 and 0.4 < smoothed[20, 1] < 0.6
        num_samples = 4000
        samples = vmap(lambda k: log_space_hmm_posterior_sample(k, pi0, jnp.log(P), ll))(jr.split(jr.PRNGKey(1), num_samples))
        freq = jnp.stack([jnp.mean(samples == k, 0) for k in range(K)], 1)
        assert jnp.allclose(freq, exact_marginals, atol=0.03)
        assert jnp.all(jnp.diff(samples, axis=1) >= 0) and jnp.all(jnp.sum(samples == 1, axis=1) == 1)

    @pytest.mark.parametrize("model", [GaussianSLDS(3, 2, 4)])
    def test_initial_posterior_is_the_uniform_chain(self, model):
        """The starting q(z) stores log transition matrices (rows of exp sum to one) and a zero log normalizer."""
        params, _ = model.initialize(jr.PRNGKey(0))
        y = jr.normal(jr.PRNGKey(1), (10, 4))
        qz = initialize_posterior(model, params, y, jnp.zeros((10, 0))).discrete
        assert jnp.allclose(jnp.exp(qz.log_transition_matrices).sum(-1), 1.0, atol=1e-6)
        assert jnp.allclose(qz.log_normalizer, 0.0, atol=1e-4)
        assert jnp.allclose(qz.smoothed_probs, 1.0 / 3)


def test_exactness_under_x64():
    """Re-run this file with 64-bit floats, where the tolerances tighten to 1e-7."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "not under_x64"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
test_exactness_under_x64.__test__ = not X64     # already under x64: nothing to re-run
