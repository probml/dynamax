import pytest

import casino_hmm

# Run the casino demo in test mode (no plotting)
def test_casino():
    casino_hmm.demo(test_mode=True)
