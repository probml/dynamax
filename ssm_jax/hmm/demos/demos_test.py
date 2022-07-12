import pytest

from ssm_jax.hmm.demos import casino_hmm, gaussian_hmm_2d, gaussian_hmm_2d_fit

# Run all the demos in test mode, which turns off plotting


def test_casino():
    casino_hmm.main(test_mode=True)


def test_gaussian():
    gaussian_hmm_2d.main(num_timesteps=100, test_mode=True)
    gaussian_hmm_2d_fit.main(num_em_iters=5, num_sgd_iters=5, num_timesteps=100, test_mode=True)
