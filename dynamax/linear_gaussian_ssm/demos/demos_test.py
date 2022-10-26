import pytest

from dynamax.linear_gaussian_ssm.demos import  kf_spiral, lgssm_learning

# Run all the demos in test mode, which turns off plotting

def test_lgssm_learning_demo():
    lgssm_learning.main(test_mode=True, method='EM')
    lgssm_learning.main(test_mode=True, method='SGD')
