import jax
from jaxtyping import Array, Float, Num
from typing import NamedTuple, Union, Dict, Any
import tensorflow_probability.substrates.jax.distributions as tfd
Distribution = tfd.Distribution

import optax
Optimizer = optax.GradientTransformation

InputSingle = Float[Array, "input_dim"]
InputSeq = Float[Array, "ntime input_dim"]
InputBatch = Float[Array, "nbatch ntime input_dim"]

StateSingleScalar = Float[Array, ""]
StateSingleVec = Float[Array, "state_dim"]
StateSingle = Union[StateSingleScalar, StateSingleVec]

StateSeqScalar = Float[Array, "ntime"]
StateSeqVec = Float[Array, "ntime state_dim"]
StateSeq = Union[StateSeqScalar, StateSeqVec]

StateBatchScalar = Float[Array, "nbatch ntime"]
StateBatchVec = Float[Array, "nbatch ntime state_dim"]
StateBatch = Union[StateBatchScalar, StateBatchVec]

EmissionSingleScalar = Float[Array, ""]
EmissionSingleVec = Float[Array, "emission_dim"]
EmissionSingle = Union[EmissionSingleScalar, EmissionSingleVec]

EmissionSeqScalar = Float[Array, "ntime"]
EmissionSeqVec = Float[Array, "ntime emission_dim"]
EmissionSeq = Union[EmissionSeqScalar, EmissionSeqVec]

EmissionBatchScalar = Float[Array, "nbatch ntime"]
EmissionBatchVec = Float[Array, "nbatch ntime emission_dim"]
EmissionBatch = Union[EmissionBatchScalar, EmissionBatchVec]


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
# Store the expected sufficient statistics from a batch of data.

LossTrace = Float[Array, "nsteps"]
# Store the sequences of losses over time from an optimizer

HMMTransitionMatrix = Float[Array, "state_dim state_dim"]
