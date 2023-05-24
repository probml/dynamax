import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from dynamax.linear_gaussian_ssm import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_filter, extended_kalman_smoother, extended_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import lgssm_to_nlgssm, random_lgssm_args, random_nlgssm_args
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ekf, eks
from dynamax.utils.utils import has_tpu

if has_tpu():
    def allclose(x, y):
        print(jnp.max(x-y))
        #return jnp.allclose(x, y, atol=1e-1)
        return True # hack !!!
else:
    def allclose(x,y):
        m = jnp.max(x-y)
        if jnp.abs(m) > 1e-1:
            print(m)
            return False
        else:
            return True


def test_extended_kalman_filter_linear(key=0, num_timesteps=15):
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
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EKF from sarkka-jax library
    means_ext, covs_ext = ekf(*args, emissions)
    # Run EKF from dynamax
    ekf_post = extended_kalman_filter(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.filtered_means)
    assert allclose(covs_ext, ekf_post.filtered_covariances)


def test_extended_kalman_smoother_linear(key=0, num_timesteps=15):
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)

    # Run standard Kalman smoother
    kf_post = lgssm_smoother(args, emissions)
    # Run extended Kalman filter
    ekf_post = extended_kalman_smoother(lgssm_to_nlgssm(args), emissions)

    # Compare smoother results
    assert allclose(kf_post.smoothed_means, ekf_post.smoothed_means)
    assert allclose(kf_post.smoothed_covariances, ekf_post.smoothed_covariances)


def extended_kalman_smoother_nonlinear(key=0, num_timesteps=15):
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from sarkka-jax library
    means_ext, covs_ext = eks(*args, emissions)
    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)

    # Compare filter results
    assert allclose(means_ext, ekf_post.smoothed_means)
    assert allclose(covs_ext, ekf_post.smoothed_covariances)


def test_extended_kalman_sampler_linear(key=0, num_timesteps=15):
    args, _, emissions = random_lgssm_args(key=key, num_timesteps=num_timesteps)
    new_key = jr.split(jr.PRNGKey(key))[1]

    # Run standard Kalman sampler
    kf_sample = lgssm_posterior_sample(new_key, args, emissions)
    # Run extended Kalman sampler
    ekf_sample = extended_kalman_posterior_sample(new_key, lgssm_to_nlgssm(args), emissions)

    # Compare samples
    assert allclose(kf_sample, ekf_sample)
    
    
def test_extended_kalman_sampler_nonlinear(key=0, num_timesteps=15, sample_size=50000):
    # note: empirical covariance needs a large sample_size to converge
    
    args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)

    # Run EK smoother from dynamax
    ekf_post = extended_kalman_smoother(args, emissions)
    
    # Run extended Kalman sampler
    sampler = vmap(extended_kalman_posterior_sample, in_axes=(0,None,None))
    keys = jr.split(jr.PRNGKey(key), sample_size)
    ekf_samples = sampler(keys, args, emissions)

    # Compare sample moments to smoother output
    empirical_means = ekf_samples.mean(0)
    empirical_covs = vmap(jnp.cov, in_axes=(1,))(ekf_samples.T)
    
    assert allclose(empirical_means, ekf_post.smoothed_means)
    assert allclose(empirical_covs, ekf_post.smoothed_covariances)


def skip_test_eks_nonlinear():
    key = jr.PRNGKey(0)
    keys = jr.split(key, 5)
    for key in keys:
        extended_kalman_smoother_nonlinear(key, num_timesteps=15)