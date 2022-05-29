import pytest

import casino_hmm
import gaussian_hmm_2d, gaussian_hmm_2d_fit

# Run all the demos in test mode, which turns off plotting

def test_casino():
    casino_hmm.demo(test_mode=True)


def test_gaussian():
    gaussian_hmm_2d.demo(num_timesteps=100, test_mode=True)
    gaussian_hmm_2d_fit.demo(num_em_iters=5, num_sgd_iters=5, num_timesteps=100, test_mode=True)
