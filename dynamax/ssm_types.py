from abc import ABC
from abc import abstractmethod


import jax
from jax import jit, lax, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jr


PRNGKey = jax.random.PRNGKey
from jaxtyping import Array, Float, PyTree, Bool, Int, Num
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable
import chex
from dataclasses import dataclass

import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
Distribution = tfd.Distribution

import optax
Optimizer = optax.GradientTransformation

#LatentType = TypeVar('LatentType', bound=ArrayTree)
#ObsType = TypeVar('ObsType', bound=ArrayTree)


InputSingle = Num[Array, "input_dim"]
InputSeq = Num[Array, "ntime input_dim"]
InputBatch = Num[Array, "nbatch ntime input_dim"]

StateSingleScalar = Num[Array, ""]
StateSingleVec = Num[Array, "state_dim"]
StateSingle = Union[StateSingleScalar, StateSingleVec]

StateSeqScalar = Num[Array, "ntime"]
StateSeqVec = Num[Array, "ntime state_dim"]
StateSeq = Union[StateSeqScalar, StateSeqVec]

StateBatchScalar = Num[Array, "nbatch ntime"]
StateBatchVec = Num[Array, "nbatch ntime state_dim"]
StateBatch = Union[StateBatchScalar, StateBatchVec]

EmissionSingleScalar = Num[Array, ""]
EmissionSingleVec = Num[Array, "emission_dim"]
EmissionSingle = Union[EmissionSingleScalar, EmissionSingleVec]

EmissionSeqScalar = Num[Array, "ntime"]
EmissionSeqVec = Num[Array, "ntime emission_dim"]
EmissionSeq = Union[EmissionSeqScalar, EmissionSeqVec]

EmissionBatchScalar = Num[Array, "nbatch ntime"]
EmissionBatchVec = Num[Array, "nbatch ntime emission_dim"]
EmissionBatch = Union[EmissionBatchScalar, EmissionBatchVec]


ParamsSSM = Dict

ParamPropsSSM = Dict

SuffStatsSSM = Any

@chex.dataclass
class PosteriorSSM(ABC):
    """Store the output of filtering or smoothing."""


SuffStatsSSM = Any
# Store the expected sufficient statistics from a batch of data.

LossTrace = Float[Array, "nsteps"]
# Store the sequences of losses over time from an optimizer 

HMMTransitionMatrix = Float[Array, "state_dim state_dim"]
