"""
Tests for the extended Kalman filter and smoother.
"""
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from dynamax.linear_gaussian_ssm import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter, extended_kalman_smoother, extended_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import lgssm_to_nlgssm, random_lgssm_args, random_nlgssm_args
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ekf, eks
from dynamax.utils.utils import has_tpu
from jax import vmap

if has_tpu():
    # TPU has very poor numerical stability
    allclose = partial(jnp.allclose, atol=1e-3)
else:
    allclose = partial(jnp.allclose, atol=1e-4)


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman filter produces the correct filtered moments
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman filter
    kf_post = lgssm_filter(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(args), emissions)

    # Compare filter results
    assert allclose(kf_post.marginal_loglik, ekf_post.marginal_loglik)
    assert allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)


def test_extended_kalman_filter_nonlinear(key=42, num_timesteps=15):
    """
    Test that the extended Kalman filter produces the correct filtered moments
    by comparing it to the sarkka-jax library.
    """
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*args, emissions)
    # Run EKF from dynamax
    ekf_post = extended_kalman_filter(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.filtered_means)
    assert allclose(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman smoother produces the correct smoothed moments
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman smoother
    kf_post = lgssm_smoother(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(lgssm_to_nlgssm(args), emissions)

    # Compare smoother results
    assert allclose(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert allclose(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman smoother produces the correct smoothed moments
    by comparing it to the sarkka-jax library.
    """
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*args, emissions)
    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.smoothed_means)
    assert allclose(covs_ext, ekf_post.smoothed_covariances)


def test_extended_kalman_sampler_linear(key=0, num_timesteps=15):
    """
    Test that the extended Kalman sampler produces samples with the correct mean
    in the linear Gaussian case.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)
    new_key = jr.split(jr.PRNGKey(key))[1]

    # Run standard Kalman sampler
    kf_sample = lgssm_posterior_sample(new_key, args, emissions)
    # Run extended Kalman sampler
    ekf_sample = extended_kalman_posterior_sample(new_key, lgssm_to_nlgssm(args), emissions)

    # Compare samples
    assert allclose(kf_sample, ekf_sample)
    
    
def test_extended_kalman_sampler_nonlinear(key=0, num_timesteps=15, sample_size=50000):
    """
    Test that the extended Kalman sampler produces samples with the correct mean.
    """
    # note: empirical covariance needs a large sample_size to converge
    
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)
    
    # Run extended Kalman sampler
    sampler = vmap(extended_kalman_posterior_sample, in_axes=(0,None,None))
    keys = jr.split(jr.PRNGKey(key), sample_size)
    ekf_samples = sampler(keys, args, emissions)

    # Compare sample moments to smoother output
    # Use the posterior variance to compute the variance of the Monte Carlo estimate,
    # and check that the differences are within 6 standard deviations.
    post_variance = vmap(jnp.diag)(ekf_post.smoothed_covariances)
    threshold = 6 * jnp.sqrt(post_variance / sample_size)
    empirical_means = ekf_samples.mean(0)
    assert jnp.all(abs(empirical_means - ekf_post.smoothed_means) < threshold)
    

def test_iterated_extended_kalman_filter_linear(key=0, num_timesteps=15):
    """
    Test that re-linearizing in the update step leaves the exact result of the
    linear Gaussian case unchanged.
    """
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)
    kf_post = lgssm_filter(args, emissions)
    ekf_post = extended_kalman_filter(lgssm_to_nlgssm(args), emissions, num_iter=4)

    assert allclose(kf_post.filtered_means, ekf_post.filtered_means)
    assert allclose(kf_post.filtered_covariances, ekf_post.filtered_covariances)
    assert allclose(kf_post.marginal_loglik, ekf_post.marginal_loglik)


def test_iterated_extended_kalman_filter_converges_to_map():
    """
    With a nonlinear emission function the iterated update is a Gauss-Newton
    solve for the posterior mode, which can be found independently on a grid.
    """
    from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM

    params = ParamsNLGSSM(
        initial_mean=jnp.array([0.5]),
        initial_covariance=jnp.array([[1.0]]),
        dynamics_function=lambda x: x,
        dynamics_covariance=jnp.array([[1.0]]),
        emission_function=lambda x: x + 0.5 * x ** 3,
        emission_covariance=jnp.array([[0.1]]),
    )
    emissions = jnp.array([[3.0]])

    grid = jnp.linspace(-3.0, 3.0, 600001)
    neg_log_post = 0.5 * (grid - 0.5) ** 2 / 1.0 + 0.5 * (3.0 - (grid + 0.5 * grid ** 3)) ** 2 / 0.1
    x_map = grid[jnp.argmin(neg_log_post)]

    post = extended_kalman_filter(params, emissions, num_iter=10)
    assert jnp.allclose(post.filtered_means[0, 0], x_map, atol=1e-3)
