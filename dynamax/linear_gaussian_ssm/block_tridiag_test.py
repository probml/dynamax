"""
Tests for the block-tridiagonal Gaussian primitives. ``test_exactness_under_x64`` re-runs
this file in a 64-bit subprocess with the tight tolerances.
"""
import os
import subprocess
import sys

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.linear_gaussian_ssm.block_tridiag import (
    BlockTridiagPotentials, block_tridiag_mvn_smoother, block_tridiag_mvn_solve, block_tridiag_mvn_matvec,
    block_tridiag_mvn_entropy, _sample_with_noise)
from dynamax.linear_gaussian_ssm.info_inference import block_tridiag_mvn_expectations

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


def test_sampler_is_the_affine_map_of_the_posterior():
    """With fixed noise the sampler is affine, x = A eps + m, with m the mean and A A^T the covariance."""
    potentials = random_potentials(jr.PRNGKey(4), 4, 2)
    mean, cov, _, _ = dense_moments(potentials)
    posterior = block_tridiag_mvn_smoother(potentials)
    f = lambda eps: _sample_with_noise(posterior, eps.reshape(4, 2)).ravel()
    A = jax.jacobian(f)(jnp.zeros(8))
    assert jnp.allclose(f(jnp.zeros(8)), mean.ravel(), atol=10 * TOL, rtol=10 * TOL)
    assert jnp.allclose(A @ A.T, cov, atol=10 * TOL, rtol=10 * TOL)


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


def test_exactness_under_x64():
    """Re-run this file with 64-bit floats, where the base tolerance tightens from 1e-4 to 1e-8."""
    if X64:
        pytest.skip("already running under x64")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "not under_x64"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-3000:]
