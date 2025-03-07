"""
Tests for the unscented Kalman filter and smoother.
"""
import jax.numpy as jnp

from dynamax.nonlinear_gaussian_ssm.inference_ukf import unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ukf, uks
from dynamax.nonlinear_gaussian_ssm.inference_test_utils import random_nlgssm_args
from dynamax.utils.utils import has_tpu
from functools import partial

if has_tpu():
    allclose = partial(jnp.allclose, atol=1e-1)
else:
    allclose = partial(jnp.allclose, atol=1e-4)

def test_ukf_nonlinear(key=0, num_timesteps=15):
    """
    Test that the unscented Kalman filter produces the correct filtered and smoothed moments
    by comparing it to the sarkka-jax library.
    """
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)
    hyperparams = UKFHyperParams()

    # Run UKF from sarkka-jax library
    means_ukf, covs_ukf = ukf(*nlgssm_args, *hyperparams, emissions)
    # Run UKS from sarkka-jax library
    means_uks, covs_uks = uks(*nlgssm_args, *hyperparams, emissions)
    # Run UKS from dynamax
    uks_post = unscented_kalman_smoother(nlgssm_args, emissions, hyperparams)

    # Compare filter results
    assert allclose(means_ukf, uks_post.filtered_means)
    assert allclose(covs_ukf, uks_post.filtered_covariances)
    assert allclose(means_uks, uks_post.smoothed_means)
    assert allclose(covs_uks, uks_post.smoothed_covariances)