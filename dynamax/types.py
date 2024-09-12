from typing import Union
from jaxtyping import Array, Float
import jax._src.random as prng


PRNGKey = prng.KeyArray

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
