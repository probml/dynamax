import chex
from typing import Callable
from functools import partial
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map

from jaxtyping import Array, Float, PyTree, Bool, Int, Num
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable

import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from dynamax.abstractions import SSM
from dynamax.linear_gaussian_ssm.inference import PosteriorLGSSMFiltered, PosteriorLGSSMSmoothed

tfd = tfp.distributions
tfb = tfp.bijectors


PosteriorNLGSSMFiltered = PosteriorLGSSMFiltered # posterior is same as linear-gaussian case
PosteriorNLGSSMSmoothed = PosteriorLGSSMSmoothed # posterior is same as linear-gaussian case

FnStateToState = Callable[[Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[[Float[Array, "state_dim input_dim"]], Float[Array, "state_dim"]]

FnStateToEmission = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[[Float[Array, "state_dim input_dim"]], Float[Array, "emission_dim"]]

@chex.dataclass
class ParamsNLGSSM:
    """Lightweight container for NLGSSM parameters."""
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]

_zeros_if_none = lambda x, shp: x if x is not None else jnp.zeros(shp)


class NonlinearGaussianSSM(SSM):
    """
    NonLinear Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)
    p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)
    p(z_1) = N(z_1 | m, S)
    where z_t = hidden, y_t = observed, u_t = inputs (can be None),
    """

    def __init__(self, state_dim: int, emission_dim: int, input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamsNLGSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsNLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        f = params.dynamics_function 
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_covariance)

    def emission_distribution(
        self,
        params: ParamsNLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_covariance)


