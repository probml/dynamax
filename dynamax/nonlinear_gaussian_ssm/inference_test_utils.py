"""
This file contains utility functions that are used to test EKF, UKF and GGF inference,
by comparing the results to the sarkka-lib codebase on some toy problems.
"""
import jax.random as jr
import jax.numpy as jnp

from jaxtyping import Array, Float
from typing import Tuple, Union
import tensorflow_probability.substrates.jax as tfp

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM, NonlinearGaussianSSM
from dynamax.parameters import ParameterProperties
from dynamax.ssm import SSM
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.types import PRNGKeyT


tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance


def lgssm_to_nlgssm(params: ParamsLGSSM) -> ParamsNLGSSM:
    """Generates NonLinearGaussianSSM params from LinearGaussianSSM params"""
    nlgssm_params = ParamsNLGSSM(
        initial_mean=params.initial.mean,
        initial_covariance=params.initial.cov,
        dynamics_function=lambda x: params.dynamics.weights @ x + params.dynamics.bias,
        dynamics_covariance=params.dynamics.cov,
        emission_function=lambda x: params.emissions.weights @ x + params.emissions.bias,
        emission_covariance=params.emissions.cov,
    )
    return nlgssm_params


def random_lgssm_args(key: Union[int, PRNGKeyT] = 0,
                      num_timesteps: int = 15,
                      state_dim: int = 4,
                      emission_dim: int = 2) -> \
                      Tuple[ParamsLGSSM, Float[Array, "ntime state_dim"], 
                            Float[Array, "ntime emission_dim"]]:
    """
    Generates random LGSSM parameters, states and emissions.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    *keys, subkey = jr.split(key, 9)

    # Generate random parameters
    initial_mean = jr.normal(keys[0], (state_dim,))
    initial_covariance = jnp.eye(state_dim) * jr.uniform(keys[1])
    dynamics_covariance = jnp.eye(state_dim) * jr.uniform(keys[2])
    emission_covariance = jnp.eye(emission_dim) * jr.uniform(keys[3])

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=initial_mean,
            cov=initial_covariance
            ),
        dynamics=ParamsLGSSMDynamics(
            weights=jr.normal(keys[4], (state_dim, state_dim)),
            bias=jr.normal(keys[5], (state_dim,)),
            input_weights=jnp.zeros((state_dim, 0)),
            cov=dynamics_covariance
            ),
        emissions=ParamsLGSSMEmissions(
            weights=jr.normal(keys[6], (emission_dim, state_dim)),
            bias=jr.normal(keys[7], (emission_dim,)),
            input_weights=jnp.zeros((emission_dim, 0)),
            cov=emission_covariance,
            )
    )

    # Generate random samples
    model = LinearGaussianSSM(state_dim, emission_dim)
    key, subkey = jr.split(subkey, 2)
    states, emissions = model.sample(params, key, num_timesteps)
    return params, states, emissions


def to_poly(state, degree):
    """
    Returns the polynomial features of the state up to the given degree.
    """
    return jnp.concatenate([state**d for d in jnp.arange(degree+1)])

def make_nlgssm_params(state_dim, 
                       emission_dim, 
                       dynamics_degree=1, 
                       emission_degree=1, 
                       key=jr.PRNGKey(0)):
    """
    Generates random NLGSSM parameters.
    """
    dynamics_weights = jr.normal(key, (state_dim, state_dim * (dynamics_degree+1)))
    f = lambda z: jnp.sin(dynamics_weights @ to_poly(z, dynamics_degree))
    emission_weights = jr.normal(key, (emission_dim, state_dim * (emission_degree+1)))
    h = lambda z: jnp.cos(emission_weights @ to_poly(z, emission_degree))
    params = ParamsNLGSSM(
        initial_mean = 0.2 * jnp.ones(state_dim),
        initial_covariance = jnp.eye(state_dim),
        dynamics_function = f,
        dynamics_covariance = jnp.eye(state_dim),
        emission_function = h,
        emission_covariance = jnp.eye(emission_dim)
    )
    return params

class SimpleNonlinearSSM(SSM):
    """
    Simple nonlinear SSM with sinusoidal dynamics and cosine emissions.
    """
    def __init__(self, state_dim, emission_dim, dynamics_degree=1, emission_degree=1):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.dynamics_degree = dynamics_degree
        self.emission_degree = emission_degree

    @property
    def emission_shape(self):
        """Returns the shape of the emission distribution."""
        return (self.emission_dim,)

    def initial_distribution(self, params, covariates=None):
        """Returns the initial distribution."""
        return MVN(params["initial"]["mean"], params["initial"]["cov"])

    def transition_distribution(self, params, state, covariates=None):
        """Returns the nonlinear dynamics function."""
        x = to_poly(state, self.dynamics_degree)
        mean = jnp.sin(params["dynamics"]["weights"] @ x)
        return MVN(mean, params["dynamics"]["cov"])

    def emission_distribution(self, params, state, covariates=None):
        """Returns the nonlinear emission function."""
        x = to_poly(state, self.emission_degree)
        mean = jnp.cos(params["emissions"]["weights"] @ x)
        return MVN(mean, params["emissions"]["cov"])

    def initialize(self, key):
        """Initializes the parameters."""
        key1, key2 = jr.split(key)
        params = dict(
            initial=dict(mean=0.2 * jnp.ones(self.state_dim), cov=jnp.eye(self.state_dim)),
            dynamics=dict(weights=jr.normal(key1, (self.state_dim, self.state_dim * (self.dynamics_degree+1))),
                          cov=jnp.eye(self.state_dim)),
            emissions=dict(weights=jr.normal(key2, (self.emission_dim, self.state_dim * (self.emission_degree+1))),
                          cov=jnp.eye(self.emission_dim)),
        )

        param_props = dict(
            initial=dict(mean=ParameterProperties(),
                         cov=ParameterProperties(constrainer=RealToPSDBijector())),
            dynamics=dict(weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=RealToPSDBijector())),
            emissions=dict(weights=ParameterProperties(),
                           cov=ParameterProperties(constrainer=RealToPSDBijector())),
        )
        return params, param_props

    def _make_inference_args(self, params):
        """Returns the inference arguments."""
        f = lambda state: jnp.sin(params["dynamics"]["weights"] @ to_poly(state, self.dynamics_degree))
        h = lambda state: jnp.cos(params["emissions"]["weights"] @ to_poly(state, self.emission_degree))
        return ParamsNLGSSM(
            initial_mean=params["initial"]["mean"],
            initial_covariance=params["initial"]["cov"],
            dynamics_function=f,
            dynamics_covariance=params["dynamics"]["cov"],
            emission_function=h,
            emission_covariance=params["emissions"]["cov"])


def random_nlgssm_args(key=0, num_timesteps=15, state_dim=4, emission_dim=2):
    """
    Generates random NLGSSM parameters, states and emissions.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    init_key, sample_key = jr.split(key, 2)
    params = make_nlgssm_params(state_dim, emission_dim, key=init_key)
    model = NonlinearGaussianSSM(state_dim, emission_dim)
    states, emissions = model.sample(params, sample_key, num_timesteps)
    return params, states, emissions

