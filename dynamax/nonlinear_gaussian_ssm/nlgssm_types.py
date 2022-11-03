
from dynamax.linear_gaussian_ssm.lgssm_types import PosteriorLGSSM

from jaxtyping import Array, Float, PyTree, Bool, Int, Num

import typing_extensions
from typing_extensions import Protocol
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable

import chex


@chex.dataclass
class ParamsNLGSSM:
    """Lightweight container for NLGSSM parameters."""
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_function: Callable
    emission_covariance: chex.Array

ParamsNLGSSMInf = ParamsNLGSSM

PosteriorNLGSSM = PosteriorLGSSM
