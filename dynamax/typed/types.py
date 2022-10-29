
from abc import ABC
from abc import abstractmethod

import jax
#https://github.com/google/jaxtyping/blob/main/jaxtyping/__init__.py
from jaxtyping import Array, Float, PyTree, Bool, Int
from typeguard import typechecked

import typing_extensions
from typing_extensions import Protocol
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping

#https://github.com/deepmind/chex/blob/master/chex/_src/pytypes.py
import chex
#from chex import Array, ArrayTree, Scalar, Numeric
#from chex import PRNGKey

#import jax._src.prng as prng
#PRNGKey = prng.PRNGKeyArray
PRNGKey = jax.random.PRNGKey

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.bijectors as tfb

#LatentType = TypeVar('LatentType', bound=ArrayTree)
#ObsType = TypeVar('ObsType', bound=ArrayTree)

InputSingle = Float[Array, "covariate_dim"]
InputSeq = Float[Array, "ntime covariate_dim"]
InputBatch = Float[Array, "nbatch ntime covariate_dim"]

StateSingleDiscrete = Int
StateSingleVec = Float[Array, "state_dim"]
StateSingle = Union[StateSingleDiscrete, StateSingleVec]

StateSeqDiscrete = Int[Array, "ntime"]
StateSeqVec = Float[Array, "ntime state_dim"]
StateSeq = Union[StateSeqDiscrete, StateSeqVec]

StateBatchDiscrete = Int[Array, "nbatch ntime"]
StateBatchVec = Float[Array, "nbatch ntime state_dim"]
StateBatch = Union[StateBatchDiscrete, StateBatchVec]

EmissionSingleDiscrete = Int
EmissionSingleVec = Float[Array, "emission_dim"]
EmissionSingle = Union[EmissionSingleDiscrete, EmissionSingleVec]

EmissionSeqDiscrete = Int[Array, "ntime"]
EmissionSeqVec = Float[Array, "ntime emission_dim"]
EmissionSeq = Union[EmissionSeqDiscrete, EmissionSeqVec]

EmissionBatchDiscrete = Int[Array, "nbatch ntime"]
EmissionBatchVec = Float[Array, "nbatch ntime emission_dim"]
EmissionBatch = Union[EmissionBatchDiscrete, EmissionBatchVec]

@dataclass
class ParameterProperties:
    trainable: bool = True
    constrainer: tfb.Bijector = tfb.Identity()


#Params = PyTree
#ParamProps = PyTree[ParameterProperties]

@dataclass
class ParamsHMM:
    initial_dist: Float[Array, "nstates"]
    transition_matrix:  Float[Array, "nstates nstates"]
    emissions: PyTree # many kinds of p(y|z) distributions
    likelihoods: Optional[Float[Array, "ntime nstates"]] = None # local evidence vector, lik(t,k) = p(y(t) | Z(t)=k)

@dataclass
class GaussDist:
    mean: Float[Array, "state_dim"]
    cov: Float[Array, "state_dim state_dim"]

@dataclass
class CondGaussDistLatent:
    cov: Float[Array, "state_dim state_dim"]
    weights: Float[Array, "state_dim state_dim"]
    bias: Optional[Float[Array, "state_dim"]] = None
    input_weights:  Optional[Float[Array, "covariates_dim state_dim"]] = None


@dataclass
class CondGaussDistObserved:
    cov: Float[Array, "emission_dim emission_dim"]
    weights: Float[Array, "emission_dim state_dim"]
    bias: Optional[Float[Array, "emission_dim"]] = None
    input_weights:  Optional[Float[Array, "covariates_dim emission_dim"]] = None

@dataclass
class ParamsLGSSM:
    initial_dist: GaussDist
    dynamics: CondGaussDistLatent
    emissions: CondGaussDistObserved

Params = Union[ParamsHMM, ParamsLGSSM]

@dataclass
class ParamPropsHMM:
    initial_dist: ParameterProperties
    transition_matrix: ParameterProperties
    emissions: Mapping[str, ParameterProperties]

@dataclass
class ParamPropsGauss:
    cov:  ParameterProperties
    mean:  ParameterProperties

@dataclass
class ParamPropsCondGauss:
    cov:  ParameterProperties
    weights:  ParameterProperties
    bias: Optional[ParameterProperties] = None
    input_weights: Optional[ParameterProperties] = None

@dataclass
class ParamPropsLGSSM:
    initial_dist: ParamPropsGauss
    dynamics: ParamPropsCondGauss
    emissions: ParamPropsCondGauss

ParamProps = Union[ParamPropsHMM, ParamPropsLGSSM]


@dataclass
class SSMPosterior(ABC):
    """Store the output of filtering or smoothing.
    The fields depend on the type of SSM, so this class will be subclassed.
    """

@dataclass
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
    def initialize_params(self, rng_key: PRNGKey) -> Tuple[Params, ParamProps]: ...

    @abstractmethod
    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> SSMPosterior: ...

    @abstractmethod
    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: Int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeq, EmissionSeq]:
        """Ancestral sampling from the joint model p(z(1:T), y(1:T) | u(1:T))."""
        raise NotImplementedError

