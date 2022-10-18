from typing import Callable
import chex


@chex.dataclass
class NLGSSMParams:
    """Lightweight container for NLGSSM parameters."""

    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_function: Callable
    emission_covariance: chex.Array
