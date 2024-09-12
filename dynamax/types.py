from typing import Union
from jaxtyping import Array, Float

PRNGKeyT = Array

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
