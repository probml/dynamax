
from abc import ABC
from abc import abstractmethod

#https://github.com/google/jaxtyping/blob/main/jaxtyping/__init__.py

from jaxtyping import Array, Float, PyTree, Bool, Int
from typeguard import typechecked

import typing_extensions
from typing_extensions import Protocol
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic

#https://github.com/deepmind/chex/blob/master/chex/_src/pytypes.py
import chex
from chex import Array, ArrayTree, Scalar, Numeric, PRNGKey

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.bijectors as tfb

EmissionSingle = Float[Array, "Nemit"]
EmissionSeq = Float[Array, "Ntime Nemit"]
EmissionBatch = Float[Array, "Nseq Ntime Nemit"]

InputSingle = Float[Array, "Ninput"]
InputSeq = Float[Array, "Ntime Ninput"]
InputBatch = Float[Array, "Nseq Ntime Ninput"]

LatentType = TypeVar('LatentType', bound=ArrayTree)
ObsType = TypeVar('ObsType', bound=ArrayTree)

StateSingleDiscrete = Int
StateSingleVec = Float[Array, "state_dim"]
StateSingle = Union[StateSingleDiscrete, StateSingleVec]

StateSeqDiscrete = Int[Array, "ntime"]
StateSeqVec = Float[Array, "ntime state_dim"]
StateSeq = Union[StateSeqDiscrete, StateSeqVec]

StateBatchDiscrete = Int[Array, "nbatch ntime"]
StateBatchVec = Float[Array, "nbatch ntime state_dim"]
StateBatch = Union[StateBatchDiscrete, StateBatchVec]


#StateSingle = Float[Array, "Nstate"]
#StateSeq = Float[Array, "Ntime Nstate"]
#StateBatch = Float[Array, "Nseq Ntime Nstate"]


@chex.dataclass
class ParameterProperties:
    trainable: bool = True
    constrainer: tfb.Bijector = tfb.Identity()

Params = PyTree
ParamProps = PyTree[ParameterProperties]


class SSMPosterior(ABC):
    """Store the output of filtering or smoothing.
    The fields depend on the type of SSM, so this class will be subclassed.
    """

@chex.dataclass
class GSSMPosterior(SSMPosterior):
    """Simple wrapper for properties of an Gaussian SSM posterior distribution.

    Attributes:
            filtered_means: (T,D_hid) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_covariances: (T,D_hid,D_hid) array,
                Cov[x_t | y_{1:t}, u_{1:t}].
            smoothed_means: (T,D_hid) array,
                E[x_t | y_{1:T}, u_{1:T}].
            smoothed_covariances: (T,D_hid,D_hid) array of smoothed marginal covariances,
                Cov[x_t | y_{1:T}, u_{1:T}].
            smoothed_cross: (T-1, D_hid, D_hid) array of smoothed cross products,
                E[x_t x_{t+1}^T | y_{1:T}, u_{1:T}].
    """
    marginal_loglik: chex.Scalar = None
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None
    smoothed_cross_covariances: chex.Array = None

@chex.dataclass
class HMMPosterior(SSMPosterior):
    """Simple wrapper for properties of an HMM posterior distribution.

    filtered_probs(t,k) = p(hidden(t)=k | obs(1:t))
    smoothed_probs(t,k) = p(hidden(t)=k | obs(1:T))
    initial_probs[i] = p(hidden(0)=i | obs(1:T))
    """
    marginal_loglik: chex.Scalar = None
    filtered_probs: chex.Array = None
    smoothed_probs: chex.Array = None
    initial_probs: chex.Array = None


class SSM(ABC):
    @abstractmethod
    def initialize_params(self, rng_key: PRNGKey) -> Tuple[Params, ParamProps]:
        """Create initial parameters at random."""
        raise NotImplementedError

    @abstractmethod
    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> SSMPosterior:
        """Compute filtered belief states."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: Int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeq, EmissionSeq]:
        """Ancestral sampling from the joint model p(z(1:T), y(1:T) | u(1:T))."""
        raise NotImplementedError

