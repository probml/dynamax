"""
Tests for the block-tridiagonal Gaussian primitives, against dense linear algebra, the
autodiff oracle in ``info_inference`` and the Kalman smoother. ``test_exactness_under_x64``
re-runs this file in a 64-bit subprocess with the tight tolerances.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap

from dynamax.linear_gaussian_ssm.block_tridiag import (
    BlockTridiagPotentials, block_tridiag_mvn_filter, block_tridiag_mvn_smoother,
    block_tridiag_mvn_sample, block_tridiag_mvn_solve, block_tridiag_mvn_matvec,
    block_tridiag_mvn_entropy, lgssm_to_block_tridiag, _sample_with_noise)
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions, lgssm_smoother)
from dynamax.linear_gaussian_ssm.inference_test_utils import make_timing_case, assert_sample_moments

X64 = jax.config.read("jax_enable_x64")
TOL = 1e-8 if X64 else 1e-4


def dense_precision(J_diag, J_lower):
    """Assemble the (TD, TD) precision matrix from its blocks."""
    num_timesteps, dim = J_diag.shape[:2]
    J = jnp.zeros((num_timesteps * dim, num_timesteps * dim), dtype=J_diag.dtype)
    for t in range(num_timesteps):
        sl = slice(t * dim, (t + 1) * dim)
        J = J.at[sl, sl].set(J_diag[t])
        if t < num_timesteps - 1:
            nxt = slice((t + 1) * dim, (t + 2) * dim)
            J = J.at[nxt, sl].set(J_lower[t])
            J = J.at[sl, nxt].set(J_lower[t].T)
    return J


def dense_moments(potentials):
    """Mean, joint covariance, log normalizer and log|J| by dense linear algebra."""
    J_diag, J_lower, h = potentials
    num_timesteps, dim = h.shape
    J = dense_precision(J_diag, J_lower)
    cov = jnp.linalg.inv(J)
    mean = cov @ h.ravel()
    log_det = jnp.linalg.slogdet(J)[1]
    log_normalizer = 0.5 * num_timesteps * dim * jnp.log(2 * jnp.pi) - 0.5 * log_det + 0.5 * h.ravel() @ mean
    return mean.reshape(num_timesteps, dim), cov, log_normalizer, log_det


def random_potentials(key, num_timesteps, dim):
    """Random SPD block-tridiagonal potentials from a block-bidiagonal factor, J = L L^T + I."""
    k1, k2 = jr.split(key)
    L = jr.normal(k1, (num_timesteps * dim, num_timesteps * dim))
    block = jnp.arange(num_timesteps * dim) // dim
    mask = (block[:, None] == block[None, :]) | (block[:, None] == block[None, :] + 1)
    L = jnp.where(mask, L, 0.0)
    J = L @ L.T + jnp.eye(num_timesteps * dim)
    J_diag = jnp.stack([J[t * dim:(t + 1) * dim, t * dim:(t + 1) * dim] for t in range(num_timesteps)])
    J_lower = jnp.stack([J[(t + 1) * dim:(t + 2) * dim, t * dim:(t + 1) * dim] for t in range(num_timesteps - 1)]) \
        if num_timesteps > 1 else jnp.zeros((0, dim, dim))
    h = jr.normal(k2, (num_timesteps, dim))
    return BlockTridiagPotentials(J_diag, J_lower, h)


def random_lgssm(key, num_timesteps, state_dim, emission_dim, input_dim, poison_first=True):
    """Random time-varying LGSSM with inputs; entry 0 of the dynamics arrays is NaN by default."""
    keys = jr.split(key, 12)

    def spd(k, n, batch):
        A = jr.normal(k, (batch, n, n))
        return A @ jnp.swapaxes(A, -1, -2) + n * jnp.eye(n)

    F = 0.5 * jr.normal(keys[0], (num_timesteps, state_dim, state_dim))
    B = jr.normal(keys[1], (num_timesteps, state_dim, input_dim))
    b = jr.normal(keys[2], (num_timesteps, state_dim))
    Q = spd(keys[3], state_dim, num_timesteps)
    if poison_first:
        F, B, b, Q = (a.at[0].set(jnp.nan) for a in (F, B, b, Q))
    H = jr.normal(keys[4], (num_timesteps, emission_dim, state_dim))
    D = jr.normal(keys[5], (num_timesteps, emission_dim, input_dim))
    d = jr.normal(keys[6], (num_timesteps, emission_dim))
    R = spd(keys[7], emission_dim, num_timesteps)
    params = ParamsLGSSM(initial=ParamsLGSSMInitial(mean=jr.normal(keys[8], (state_dim,)), cov=spd(keys[9], state_dim, 1)[0]),
                         dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                         emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R))
    inputs = jr.normal(keys[10], (num_timesteps, input_dim))
    emissions = jr.normal(keys[11], (num_timesteps, emission_dim))
    return params, emissions, inputs


def check_against_dense(potentials, posterior, tol):
    """Compare a BlockTridiagPosterior with dense linear algebra."""
    mean, cov, log_normalizer, log_det = dense_moments(potentials)
    num_timesteps, dim = potentials.h.shape
    assert jnp.allclose(posterior.log_normalizer, log_normalizer, atol=tol, rtol=tol)
    assert jnp.allclose(posterior.log_det_precision, log_det, atol=tol, rtol=tol)
    assert jnp.allclose(posterior.smoothed_means, mean, atol=tol, rtol=tol)
    for t in range(num_timesteps):
        sl = slice(t * dim, (t + 1) * dim)
        assert jnp.allclose(posterior.smoothed_covariances[t], cov[sl, sl], atol=tol, rtol=tol)
        if t < num_timesteps - 1:
            nxt = slice((t + 1) * dim, (t + 2) * dim)
            cross = cov[sl, nxt] + jnp.outer(mean[t], mean[t + 1])
            assert jnp.allclose(posterior.smoothed_cross_covariances[t], cross, atol=tol, rtol=tol)


@pytest.mark.parametrize("num_timesteps", [1, 2, 7])
@pytest.mark.parametrize("dim", [1, 3])
def test_smoother_matches_dense(num_timesteps, dim):
    """Log normalizer, log|J|, means, covariances and cross-moments against dense linear algebra."""
    potentials = random_potentials(jr.PRNGKey(num_timesteps * 10 + dim), num_timesteps, dim)
    check_against_dense(potentials, block_tridiag_mvn_smoother(potentials), 10 * TOL)


def test_lgssm_to_block_tridiag_hand_expanded_case():
    """The scalar three-step model of the former ``lds_to_block_tridiag`` test: expanding
    z0^2 + (z1 - 2 z0 - 10)^2 / 11 + (z2 - 3 z1 + 7)^2 / 13 + sum(z_t^2) gives the blocks below.
    Entry 0 of every dynamics array is NaN and must never be read."""
    nan = jnp.nan
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=jnp.zeros(1), cov=jnp.eye(1)),
        dynamics=ParamsLGSSMDynamics(weights=jnp.array([nan, 2., 3.])[:, None, None], bias=None,
                                     input_weights=jnp.array([nan, 5., 7.])[:, None, None],
                                     cov=jnp.array([nan, 11., 13.])[:, None, None]),
        emissions=ParamsLGSSMEmissions(weights=jnp.eye(1), bias=None, input_weights=jnp.zeros((1, 1)), cov=jnp.eye(1)))
    potentials, _ = lgssm_to_block_tridiag(params, jnp.zeros((3, 1)), jnp.array([[100.], [2.], [-1.]]))
    assert jnp.allclose(potentials.J_diag[:, 0, 0], jnp.array([2 + 4 / 11, 1 + 1 / 11 + 9 / 13, 1 + 1 / 13]), atol=1e-6)
    assert jnp.allclose(potentials.J_lower[:, 0, 0], jnp.array([-2 / 11, -3 / 13]), atol=1e-6)
    assert jnp.allclose(potentials.h[:, 0], jnp.array([-20 / 11, 10 / 11 + 21 / 13, -7 / 13]), atol=1e-6)


@pytest.mark.parametrize("num_timesteps", [1, 5])
def test_smoother_matches_autodiff_oracle(num_timesteps, dim=2):
    """The explicit backward pass agrees with differentiating the log normalizer."""
    potentials = random_potentials(jr.PRNGKey(3), num_timesteps, dim)
    posterior = block_tridiag_mvn_smoother(potentials)
    log_normalizer, Ex, ExxT, ExxnT = block_tridiag_mvn_expectations(*potentials)
    assert jnp.allclose(posterior.log_normalizer, log_normalizer, atol=10 * TOL, rtol=10 * TOL)
    assert jnp.allclose(posterior.smoothed_means, Ex, atol=10 * TOL, rtol=10 * TOL)
    outer = jnp.einsum("ti,tj->tij", posterior.smoothed_means, posterior.smoothed_means)
    assert jnp.allclose(posterior.smoothed_covariances + outer, ExxT, atol=10 * TOL, rtol=10 * TOL)
    assert jnp.allclose(posterior.smoothed_cross_covariances.swapaxes(-1, -2), ExxnT, atol=10 * TOL, rtol=10 * TOL)


@pytest.mark.parametrize("num_timesteps", [1, 2, 6])
def test_lgssm_to_block_tridiag_matches_kalman_smoother(num_timesteps):
    """Random time-varying LGSSM with inputs and NaN in the unused entry 0 of the dynamics."""
    params, emissions, inputs = random_lgssm(jr.PRNGKey(num_timesteps), num_timesteps, 2, 3, 2)
    reference = lgssm_smoother(params, emissions, inputs)
    potentials, log_constant = lgssm_to_block_tridiag(params, emissions, inputs)
    posterior = block_tridiag_mvn_smoother(potentials)
    tol = 100 * TOL
    assert jnp.allclose(posterior.smoothed_means, reference.smoothed_means, atol=tol, rtol=tol)
    assert jnp.allclose(posterior.smoothed_covariances, reference.smoothed_covariances, atol=tol, rtol=tol)
    assert jnp.allclose(posterior.smoothed_cross_covariances, reference.smoothed_cross_covariances, atol=tol, rtol=tol)
    assert jnp.allclose(posterior.log_normalizer + log_constant, reference.marginal_loglik, atol=tol, rtol=tol)


@pytest.mark.parametrize("num_timesteps", [1, 3])
def test_lgssm_to_block_tridiag_exact_case(num_timesteps):
    """Closed-form moments and marginal log likelihood of the scalar timing case."""
    params, emissions, inputs, expected = make_timing_case(num_timesteps)
    potentials, log_constant = lgssm_to_block_tridiag(params, emissions, inputs)
    posterior = block_tridiag_mvn_smoother(potentials)
    tol = 1e-8 if X64 else 2e-5
    assert jnp.allclose(posterior.smoothed_means, expected["smoothed_means"], atol=tol, rtol=tol)
    assert jnp.allclose(posterior.smoothed_covariances, expected["smoothed_covariances"], atol=tol, rtol=tol)
    assert jnp.allclose(posterior.log_normalizer + log_constant, expected["marginal_loglik"], atol=tol, rtol=tol)
    if num_timesteps > 1:
        assert jnp.allclose(posterior.smoothed_cross_covariances, expected["smoothed_cross_covariances"], atol=tol, rtol=tol)
    else:
        assert jnp.allclose(posterior.log_normalizer + log_constant, -0.5 * jnp.log(4 * jnp.pi), atol=tol)
        assert posterior.smoothed_cross_covariances.shape == (0, 1, 1)


def test_lgssm_to_block_tridiag_rejects_short_dynamics():
    """Time-varying dynamics arrays must have one entry per timestep."""
    params, emissions, inputs, _ = make_timing_case(3)
    params = params._replace(dynamics=params.dynamics._replace(cov=params.dynamics.cov[:-1]))
    with pytest.raises(AssertionError):
        lgssm_to_block_tridiag(params, emissions, inputs)


def test_sampler_is_the_affine_map_of_the_posterior():
    """With fixed noise the sampler is affine, x = A eps + m, with m the mean and A A^T the covariance."""
    potentials = random_potentials(jr.PRNGKey(4), 4, 2)
    mean, cov, _, _ = dense_moments(potentials)
    f = lambda eps: _sample_with_noise(potentials, eps.reshape(4, 2)).ravel()
    A = jax.jacobian(f)(jnp.zeros(8))
    assert jnp.allclose(f(jnp.zeros(8)), mean.ravel(), atol=10 * TOL, rtol=10 * TOL)
    assert jnp.allclose(A @ A.T, cov, atol=10 * TOL, rtol=10 * TOL)


def test_sampler_moments(num_samples=20000):
    """Whitened moments of many samples match the dense mean and covariance."""
    potentials = random_potentials(jr.PRNGKey(5), 5, 2)
    mean, cov, _, _ = dense_moments(potentials)
    keys = jr.split(jr.PRNGKey(6), num_samples)
    samples = jit(vmap(lambda k: block_tridiag_mvn_sample(k, potentials)))(keys)
    assert_sample_moments(samples, mean, cov)
    whitened = np.linalg.solve(np.linalg.cholesky(cov), (np.asarray(samples).reshape(num_samples, -1) - np.asarray(mean).ravel()).T)
    assert abs(whitened.mean(axis=1).sum()) < 3 * np.sqrt(whitened.shape[0] / num_samples)

    # The cached filtered potentials give identical samples.
    _, _, filtered_precisions, filtered_linear_potentials = block_tridiag_mvn_filter(*potentials)
    sample = block_tridiag_mvn_sample(keys[0], potentials)
    sample_cached = block_tridiag_mvn_sample(keys[0], potentials, filtered=(filtered_precisions, filtered_linear_potentials))
    assert jnp.allclose(sample, sample_cached, atol=10 * TOL)

    # Exact LGSSM case: the joint covariance of the samples.
    params, emissions, inputs, expected = make_timing_case(3)
    exact_potentials, _ = lgssm_to_block_tridiag(params, emissions, inputs)
    samples = jit(vmap(lambda k: block_tridiag_mvn_sample(k, exact_potentials)))(keys)
    assert_sample_moments(samples, expected["smoothed_means"], expected["joint_covariance"])


def test_solve_matvec_and_entropy():
    """Round trips of solve and matvec; entropy against the dense formula and the trace formula."""
    potentials = random_potentials(jr.PRNGKey(7), 6, 3)
    J_diag, J_lower, h = potentials
    v = jr.normal(jr.PRNGKey(8), h.shape)
    assert jnp.allclose(block_tridiag_mvn_matvec(J_diag, J_lower, block_tridiag_mvn_solve(J_diag, J_lower, v)), v, atol=100 * TOL, rtol=100 * TOL)
    assert jnp.allclose(block_tridiag_mvn_solve(J_diag, J_lower, block_tridiag_mvn_matvec(J_diag, J_lower, v)), v, atol=100 * TOL, rtol=100 * TOL)
    J = dense_precision(J_diag, J_lower)
    assert jnp.allclose(block_tridiag_mvn_matvec(J_diag, J_lower, v), (J @ v.ravel()).reshape(v.shape), atol=10 * TOL, rtol=10 * TOL)

    posterior = block_tridiag_mvn_smoother(potentials)
    entropy = block_tridiag_mvn_entropy(posterior.log_det_precision, *h.shape)
    _, cov, _, _ = dense_moments(potentials)
    dense_entropy = 0.5 * jnp.linalg.slogdet(2 * jnp.pi * jnp.e * cov)[1]
    assert jnp.allclose(entropy, dense_entropy, atol=100 * TOL, rtol=100 * TOL)
    # ssm's trace formula: -E_q[log q(x)] from the expected sufficient statistics.
    log_normalizer, Ex, ExxT, ExxnT = block_tridiag_mvn_expectations(*potentials)
    trace_entropy = 0.5 * jnp.sum(J_diag * ExxT) + jnp.sum(J_lower * ExxnT) - jnp.sum(h * Ex) + log_normalizer
    assert jnp.allclose(entropy, trace_entropy, atol=100 * TOL, rtol=100 * TOL)


def test_jit_and_vmap():
    """The smoother is jittable and batchable."""
    keys = jr.split(jr.PRNGKey(9), 3)
    batch = vmap(lambda k: random_potentials(k, 4, 2))(keys)
    batched = jit(vmap(block_tridiag_mvn_smoother))(batch)
    for i in range(3):
        single = block_tridiag_mvn_smoother(jax.tree_util.tree_map(lambda x: x[i], batch))
        assert jnp.allclose(batched.smoothed_means[i], single.smoothed_means, atol=10 * TOL)
        assert jnp.allclose(batched.log_normalizer[i], single.log_normalizer, atol=10 * TOL)


def test_stiff_float32_case():
    """Information-form conditioning with Q = 1e-4 and F ~ I over T = 500 steps stays finite and
    close to the moment-form Kalman smoother; reports the error with and without jitter."""
    num_timesteps, state_dim, emission_dim = 500, 2, 3
    angle = 0.01
    F = 0.999 * jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])
    key_H, key_y = jr.split(jr.PRNGKey(10))
    params = ParamsLGSSM(initial=ParamsLGSSMInitial(mean=jnp.zeros(state_dim), cov=jnp.eye(state_dim)),
                         dynamics=ParamsLGSSMDynamics(weights=F, bias=None, input_weights=None, cov=1e-4 * jnp.eye(state_dim)),
                         emissions=ParamsLGSSMEmissions(weights=jr.normal(key_H, (emission_dim, state_dim)), bias=None,
                                                        input_weights=None, cov=1e-2 * jnp.eye(emission_dim)))
    emissions = jr.normal(key_y, (num_timesteps, emission_dim))
    reference = lgssm_smoother(params, emissions)
    potentials, _ = lgssm_to_block_tridiag(params, emissions)
    for jitter in (0.0, 1e-6):
        posterior = block_tridiag_mvn_smoother(potentials, jitter=jitter)
        assert jnp.all(jnp.isfinite(posterior.smoothed_means))
        assert jnp.all(jnp.isfinite(posterior.smoothed_covariances))
        scale = jnp.abs(reference.smoothed_means).max()
        error = jnp.abs(posterior.smoothed_means - reference.smoothed_means).max() / scale
        print(f"stiff case, jitter={jitter}: max relative error of means vs lgssm_smoother = {error:.2e}")
        assert error < 1e-2


def test_exactness_under_x64():
    """Re-run this file with 64-bit floats, where the base tolerance tightens from 1e-4 to 1e-8."""
    if X64:
        pytest.skip("already running under x64")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "not under_x64"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
