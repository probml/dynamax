import pytest

from ssm_jax.lgssm.demos import (kf_tracking, kf_spiral, kf_parallel)

# Run all the demos in test mode, which turns off plotting

def test_all():
    kf_tracking.main(test_mode=True)
    kf_spiral.main(test_mode=True)
    kf_parallel.main(test_mode=True)


