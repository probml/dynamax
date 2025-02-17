"""
Nonlinear Gaussian State Space Model objects.
"""
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

from dynamax.ssm import SSM
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import NamedTuple, Optional, Union, Callable


FnStateToState = Callable[ [Float[Array, " state_dim"]], Float[Array, " state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, " state_dim"], Float[Array, " input_dim"]], Float[Array, " state_dim"]]
FnStateToEmission = Callable[ [Float[Array, " state_dim"]], Float[Array, " emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, " state_dim"], Float[Array, " input_dim"] ], Float[Array, " emission_dim"]]


class ParamsNLGSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """

    initial_mean: Float[Array, " state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]


class NonlinearGaussianSSM(SSM):
    """
    Nonlinear Gaussian State Space Model.

    The model is defined as follows

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    where the model parameters are

    * $z_t$ = hidden variables of size `state_dim`,
    * $y_t$ = observed variables of size `emission_dim`
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).
    * $f$ = dynamics (transition) function
    * $h$ = emission (observation) function
    * $Q$ = covariance matrix of dynamics (system) noise
    * $R$ = covariance matrix for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state


    These parameters of the model are stored in a separate object of type :class:`ParamsNLGSSM`.
    """
    def __init__(self, state_dim: int, emission_dim: int, input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        """Returns the shape of the emission distribution."""
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        """Returns the shape of the input distribution."""
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(self,
                             params: ParamsNLGSSM,
                             inputs: Optional[Float[Array, " input_dim"]] = None) \
                             -> tfd.Distribution:
        """Returns the initial distribution."""
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(self,
                                params: ParamsNLGSSM,
                                state: Float[Array, " state_dim"],
                                inputs: Optional[Float[Array, " input_dim"]] = None) \
                                -> tfd.Distribution:
        """Returns the nonlinear dynamics distribution."""
        f = params.dynamics_function
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_covariance)

    def emission_distribution(self,
                              params: ParamsNLGSSM,
                              state: Float[Array, " state_dim"],
                              inputs: Float[Array, " input_dim"] = None) \
                              -> tfd.Distribution:
        """Returns the nonlinear emission distribution."""
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_covariance)
