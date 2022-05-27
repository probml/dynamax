import pytest

import casino_hmm
import gaussian_hmm

def test_casino():
    # Run the casino demo in test mode (no plotting)
    casino_hmm.demo(test_mode=True)


def test_gaussian():
    # Run the casino demo in test mode (no plotting)
    gaussian_hmm.demo(num_em_iters=5, num_sgd_iters=5, num_timesteps=100, test_mode=True)
