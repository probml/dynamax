import chex
from typing import Callable
from jax import numpy as jnp
from dynamax.abstractions import SSM
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

tfd = tfp.distributions
tfb = tfp.bijectors

@chex.dataclass
class GGSSMParams:
    """Lightweight container for GGSSM parameters, used by inference algorithms."""

    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_mean_function: Callable
    emission_cov_function: Callable
    emission_dist: Callable


class GGSSM(SSM):
    """
    General Gaussian State Space Model is defined as follows:
    p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)
    p(y_t | z_t) = N(y_t | h(z_t, u_t), R(z_t, u_t))
    p(z_1) = N(z_1 | m, S)
    where z_t = hidden, y_t = observed, u_t = inputs (can be None),
    f = params.dynamics_function
    Q = params.dynamics_covariance 
    h = params.emission_mean_function
    R = params.emission_cov_function
    m = params.initial_mean
    S = params.initial_covariance
    """

    def __init__(self, state_dim, emission_dim, input_dim=0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(self, params, covariates=None):
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(self, params, state, covariates=None):
        f = params.dynamics_function
        if covariates is None:
            mean = f(state)
        else:
            mean = f(state, covariates)
        return MVN(mean, params.dynamics_covariance)

    def emission_distribution(self, params, state, covariates=None):
        h = params.emission_mean_function
        R = params.emission_cov_function
        if covariates is None:
            mean = h(state)
            cov = R(state)
        else:
            mean = h(state, covariates)
            cov = R(state, covariates)
        return params.emission_dist(mean, cov)