import chex
from typing import Callable
from functools import partial
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import tree_map
from dynamax.abstractions import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample, LGSSMParams
from dynamax.parameters import ParameterProperties
from dynamax.utils import PSDToRealBijector
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

tfd = tfp.distributions
tfb = tfp.bijectors

_zeros_if_none = lambda x, shp: x if x is not None else jnp.zeros(shp)

@chex.dataclass
class NLGSSMParams:
    """Lightweight container for NLGSSM parameters, used by inference algorithms."""

    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_function: Callable
    emission_covariance: chex.Array



class NLGSSM(SSM):
    """
    NonLinear Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)
    p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)
    p(z_1) = N(z_1 | m, S)
    where z_t = hidden, y_t = observed, u_t = inputs (can be None),
    f = params["dynamics"]["function"]
    Q = params["dynamics"]["cov"]
    h = params["emissions"]["function"]
    R = params["emissions"]["cov"]
    m = params["initial"]["mean"]
    S = params["initial"]["cov"]
    """

    def __init__(self, state_dim, emission_dim,   input_dim=0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(self, params, inputs=None):
        return MVN(params["initial"]["mean"], params["initial"]["cov"])

    def transition_distribution(self, params, state, inputs=None):
        f = params["dynamics"]["function"]
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params["dynamics"]["cov"])

    def emission_distribution(self, params, state, inputs=None):
        h = params["emissions"]["function"]
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params["emissions"]["cov"])

    def make_inference_args(self, params):
        return NLGSSMParams(
            initial_mean=params["initial"]["mean"],
            initial_covariance=params["initial"]["cov"],
            dynamics_function=params["dynamics"]["function"],
            dynamics_covariance=params["dynamics"]["cov"],
            emission_function=params["emissions"]["function"],
            emission_covariance=params["emissions"]["cov"])
