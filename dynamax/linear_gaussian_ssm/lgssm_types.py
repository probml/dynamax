from dynamax.ssm_types import *


from jaxtyping import Array, Float, PyTree, Bool, Int, Num

import typing_extensions
from typing_extensions import Protocol
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable

import chex

ParamsLGSSM = ParamsSSM

ParamPropsLGSSM = ParamPropsSSM

SuffStatsLGSSM = SuffStatsSSM

@chex.dataclass
class ParamsLGSSMInf:
    """Lightweight container for passing LGSSM parameters to inference algorithms."""
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_weights: Float[Array, "state_dim state_dim"]
    dynamics_covariance:  Float[Array, "state_dim state_dim"]
    emission_weights:  Float[Array, "emission_dim state_dim"]
    emission_covariance: Float[Array, "emission_dim emission_dim"]

    # Optional parameters (None means zeros)
    dynamics_input_weights: Optional[Float[Array, "input_dim state_dim"]] = None
    dynamics_bias: Optional[Float[Array, "state_dim"]] = None
    emission_input_weights: Optional[Float[Array, "input_dim emission_dim"]] = None
    emission_bias: Optional[Float[Array, "emission_dim"]] = None


@chex.dataclass
class PosteriorLGSSM:
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
    marginal_loglik: Optional[Float[Array, ""]] = None # Scalar
    filtered_means: Optional[Float[Array, "ntime state_dim"]] = None
    filtered_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None
    smoothed_means: Optional[Float[Array, "ntime state_dim"]] = None
    smoothed_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None
    smoothed_cross_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None

@chex.dataclass
class PosteriorLGSSMOld:
    """Simple wrapper for properties of an LGSSM posterior distribution.

    Attributes:
            marginal_loglik: marginal log likelihood of the data
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