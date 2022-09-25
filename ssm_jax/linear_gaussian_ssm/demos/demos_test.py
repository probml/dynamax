import pytest

from ssm_jax.linear_gaussian_ssm.demos import kf_tracking, kf_spiral, kf_parallel, kf_linreg, lgssm_learning

# Run all the demos in test mode, which turns off plotting
def test_kf_tracking_demo():
    kf_tracking.main(test_mode=True)


def test_kf_spiral_demo():
    kf_spiral.main(test_mode=True)


def test_kf_parallel_demo():
    kf_parallel.main(test_mode=True)


def test_kf_linreg_demo():
    kf_linreg.main(test_mode=True)


def test_lgssm_learning_demo():
    lgssm_learning.main(test_mode=True, method='EM')
    lgssm_learning.main(test_mode=True, method='SGD')
