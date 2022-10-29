
from abc import ABC
from abc import abstractmethod

from jaxtyping import Array, Float, PyTree, Bool, Int
#https://github.com/google/jaxtyping/blob/main/jaxtyping/__init__.py

from typeguard import typechecked
import typing_extensions
from typing_extensions import Protocol
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import chex
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr

import tensorflow_probability.substrates.jax.bijectors as tfb

Key = jr.PRNGKey

EmissionSingle = Float[Array, "Nemit"]
EmissionSeq = Float[Array, "Ntime Nemit"]
EmissionBatch = Float[Array, "Nseq Ntime Nemit"]

InputSingle = Float[Array, "Ninput"]
InputSeq = Float[Array, "Ntime Ninput"]
InputBatch = Float[Array, "Nseq Ntime Ninput"]

StateSingle = Float[Array, "Nstate"]
StateSeq = Float[Array, "Ntime Nstate"]
StateBatch = Float[Array, "Nseq Ntime Nstate"]


@chex.dataclass
class ParameterProperties:
    trainable: bool = True
    constrainer: tfb.Bijector = tfb.Identity()

Params = PyTree
ParamProps = PyTree[ParameterProperties]


@dataclass
class SSMPosterior:
    """Store the output of filtering or smoothing.
    The fields depend on the type of SSM, so this class will be subclassed.
    
    Attributes:
        marginal_loglik: marginal log likelihood of the data, log sum_{hidden(1:t)} prob(hidden(1:t), obs(1:t) | params)

    """
    marginal_loglik: chex.Scalar = None

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
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None
    smoothed_cross_covariances: chex.Array = None

class HMMPosterior(SSMPosterior):
    """Simple wrapper for properties of an HMM posterior distribution.

    filtered_probs(t,k) = p(hidden(t)=k | obs(1:t))
    smoothed_probs(t,k) = p(hidden(t)=k | obs(1:T))
    initial_probs[i] = p(hidden(0)=i | obs(1:T))
    """
    filtered_probs: chex.Array = None
    smoothed_probs: chex.Array = None
    initial_probs: chex.Array = None


class SSM(ABC):
    @abstractmethod
    def initialize_params(self, rng_key: Key) -> Tuple[Params, ParamProps]:
        """Create initial parameters at random."""
        raise NotImplementedError

    @abstractmethod
    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> SSMPosterior:
        """Compute filtered belief states."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: Params, rng_key: Key, num_timesteps: Int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeq, EmissionSeq]:
        """Ancestral sampling from the joint model p(z(1:T), y(1:T) | u(1:T))."""
        raise NotImplementedError


class MyLGSSM(SSM):
    def __init__(self,
                 state_dim: int,
                 emission_dim: int,
                 input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    def initialize_params(self, rng_key: Key) -> Tuple[Params, ParamProps]:
        del rng_key
        Nz, Ny = self.state_dim, self.emission_dim
        params = dict(
            initial=dict(probs=jnp.ones(Nz) / (Nz*1.0)),
            transitions=dict(transition_matrix=0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz),
            emissions=dict(means=jnp.zeros((Nz, Ny)), scales=jnp.ones((Nz, Ny)))
        )
        param_props = dict(
            initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
        )
        return (params, param_props)


    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> GSSMPosterior:
        print('filtering, nlatents ', self.state_dim)
        print('params ', params['initial']['probs'])
        loglik = jnp.sum(emissions)
        return GSSMPosterior(marginal_loglik = loglik, filtered_means = 42)

    def sample(self, params: Params, rng_key: Key, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeq, EmissionSeq]:
        states = jnp.zeros((num_timesteps, self.state_dim))
        emissions = jnp.zeros((num_timesteps, self.emission_dim))
        return (states, emissions)
