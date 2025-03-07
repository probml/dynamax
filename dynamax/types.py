"""
This module defines type aliases for the dynamax package.
"""
from typing import Union
from jaxtyping import Array, Float, Int

PRNGKeyT = Array

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float

IntScalar = Union[int, Int[Array, ""]]
