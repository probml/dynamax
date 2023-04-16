from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

from dynamax.ssm import SSM
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.slds.inference import ParamsSLDS
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update
from dynamax.utils.utils import pytree_stack, psd_solve

class SLDS(SSM):
    

    def __init__(
        self,
        num_states: int,
        state_dim: int,
        emission_dim: int,
        input_dim: int=1
    ):
        self.num_states = num_states
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamsSLDS,
        dstate = int
    ) -> tfd.Distribution:
        params = params.linear_gaussian
        return MVN(params.initial_mean[dstate], params.initial_cov[dstate])

    def transition_distribution(
        self,
        params: ParamsSLDS,
        dstate: int,
        cstate: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        params = params.linear_gaussian
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        dynamics_input_weights = params.dynamics_input_weights if params.dynamics_input_weights is not None else jnp.zeros((self.num_states, self.state_dim, self.input_dim))
        mean = params.dynamics_weights[dstate] @ cstate + dynamics_input_weights[dstate] @ inputs + params.dynamics_bias[dstate]
        return MVN(mean, params.dynamics_cov[dstate])

    def emission_distribution(
        self,
        params: ParamsSLDS,
        dstate: int,
        cstate: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        params = params.linear_gaussian
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        emission_input_weights = params.emission_input_weights if params.emission_input_weights is not None else jnp.zeros((self.num_states, self.emission_dim, self.input_dim))
        mean = params.emission_weights[dstate] @ cstate + emission_input_weights[dstate] @ inputs + params.emission_bias[dstate]
        return MVN(mean, params.emission_cov[dstate])

    def sample(
        self,
        params: ParamsSLDS,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
            Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """

        if not params.linear_gaussian.initialized: raise ValueError("ParamsSLDS must be initialized")


        def _step(prev_states, args):
            key, inpt = args
            key0, key1, key2 = jr.split(key, 3)
            dstate, cstate = prev_states
            dstate = jr.choice(key0, jnp.arange(self.num_states), p = params.discrete.transition_matrix[dstate,:])
            cstate = self.transition_distribution(params, dstate, cstate, inpt).sample(seed=key2)
            emission = self.emission_distribution(params, dstate, cstate, inpt).sample(seed=key1)
            return (dstate, cstate), (dstate, cstate, emission)

        # Sample the initial state
        key0, key1, key2, key = jr.split(key, 4)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_dstate = jr.choice(key0, jnp.arange(self.num_states), p = params.discrete.initial_distribution)
        initial_cstate = self.initial_distribution(params, initial_dstate).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_dstate, initial_cstate, initial_input).sample(seed=key2)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_dstates, next_cstates, next_emissions) = lax.scan(_step, (initial_dstate, initial_cstate), (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        dstates = tree_map(expand_and_cat, initial_dstate, next_dstates)
        cstates = tree_map(expand_and_cat, initial_cstate, next_cstates)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return dstates, cstates, emissions