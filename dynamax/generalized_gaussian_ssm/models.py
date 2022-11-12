
from jaxtyping import Array, Float
import tensorflow_probability.substrates.jax as tfp
from typing import NamedTuple, Optional, Union, Callable

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance

from dynamax.ssm import SSM
from dynamax.nonlinear_gaussian_ssm.models import FnStateToState, FnStateAndInputToState
from dynamax.nonlinear_gaussian_ssm.models import FnStateToEmission, FnStateAndInputToEmission

FnStateToEmission2 = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim emission_dim"]]
FnStateAndInputToEmission2 = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "emission_dim emission_dim"]]

# emission distribution takes a mean vector and covariance matrix and returns a distribution
EmissionDistFn = Callable[ [Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], tfd.Distribution]


class ParamsGGSSM(NamedTuple):
    """
    Container for GGSSM parameters. Differs from NLGSSM in terms of emission model.
    
    The tuple doubles as a container for the ParameterProperties.

    $$p(x_t | x_{t-1}, u_t) = N(x_t | f(x_{t-1}, u_t), Q_t)$$
    $$p(y_t | x_t) = q(y_t | h(x_t, u_t), R(x_t, u_t))$$
    $$p(z_1) = N(x_1 | m, S)$$

    :param initial_mean: $m$
    :param initial_covariance: $S$
    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_mean_function: $h$
    :param emissions_cov_function: $R$
    :param emission_dist: the observation pdf $q$. Constructed from specified mean and covariance.

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.
    """

    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance: Float[Array, "state_dim state_dim"]

    emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2]
    emission_dist: EmissionDistFn = MVN



class GeneralizedGaussianSSM(SSM):
    """
    Generalized Gaussian State Space Model.

    The model is defined as follows

    $$p(x_t | x_{t-1}, u_t) = N(x_t | f(x_{t-1}, u_t), Q_t)$$
    $$p(y_t | x_t) = q(y_t | h(x_t, u_t), R(x_t, u_t))$$
    $$p(z_1) = N(x_1 | m, S)$$

    where

    * $x_t$ = hidden variables of size `state_dim`,
    * $y_t$ = observed variables of size `emission_dim`
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).
    * $f$ = dynamics (transition) function
    * $h$ = emission (observation) function
    * $Q$ = covariance matrix of dynamics (system) noise
    * $R$ = covariance function for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

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

    def initial_distribution(
        self,
        params: ParamsGGSSM,
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsGGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        f = params.dynamics_function
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_covariance)

    def emission_distribution(
        self,
        params: ParamsGGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
    ) -> tfd.Distribution:
        h = params.emission_mean_function
        R = params.emission_cov_function
        if inputs is None:
            mean = h(state)
            cov = R(state)
        else:
            mean = h(state, inputs)
            cov = R(state, inputs)
        return params.emission_dist(mean, cov)