"""
This module contains functions for inference in linear Gaussian state space models (LGSSMs).
"""
import jax.numpy as jnp
import jax.random as jr
import inspect
import warnings

from functools import wraps
from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Array, Float
from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.parameters import ParameterProperties
from dynamax.types import PRNGKeyT, Scalar
from typing import NamedTuple, Optional, Union, Tuple

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiagPlusLowRankCovariance as MVNLowRank,
    MultivariateNormalFullCovariance as MVN)

class ParamsLGSSMInitial(NamedTuple):
    r"""Parameters of the initial distribution

    $$p(z_1) = \mathcal{N}(z_1 \mid \mu_1, Q_1)$$

    The tuple doubles as a container for the ParameterProperties.

    :param mean: $\mu_1$
    :param cov: $Q_1$

    """
    mean: Union[Float[Array, " state_dim"], ParameterProperties]
    # unconstrained parameters are stored as a vector.
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, " state_dim_triu"], ParameterProperties]


class ParamsLGSSMDynamics(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(z_{t+1} \mid z_t, u_t) = \mathcal{N}(z_{t+1} \mid F z_t + B u_t + b, Q)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: dynamics weights $F$
    :param bias: dynamics bias $b$
    :param input_weights: dynamics input weights $B$
    :param cov: dynamics covariance $Q$

    """
    weights: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"]]

    bias: Union[ParameterProperties,
        Float[Array, " state_dim"],
        Float[Array, "ntime state_dim"]]

    input_weights: Union[ParameterProperties,
        Float[Array, "state_dim input_dim"],
        Float[Array, "ntime state_dim input_dim"]]

    cov: Union[ParameterProperties,
        Float[Array, "state_dim state_dim"],
        Float[Array, "ntime state_dim state_dim"],
        Float[Array, " state_dim_triu"]]


class ParamsLGSSMEmissions(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(y_t \mid z_t, u_t) = \mathcal{N}(y_t \mid H z_t + D u_t + d, R)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: emission weights $H$
    :param bias: emission bias $d$
    :param input_weights: emission input weights $D$
    :param cov: emission covariance $R$

    """
    weights: Union[ParameterProperties,
        Float[Array, "emission_dim state_dim"],
        Float[Array, "ntime emission_dim state_dim"]]

    bias: Union[ParameterProperties,
        Float[Array, " emission_dim"],
        Float[Array, "ntime emission_dim"]]

    input_weights: Union[ParameterProperties,
        Float[Array, "emission_dim input_dim"],
        Float[Array, "ntime emission_dim input_dim"]]

    cov: Union[ParameterProperties,
        Float[Array, "emission_dim emission_dim"],
        Float[Array, "ntime emission_dim emission_dim"],
        Float[Array, " emission_dim"],
        Float[Array, "ntime emission_dim"],
        Float[Array, " emission_dim_triu"]]


class ParamsLGSSM(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsLGSSMDynamics
    emissions: ParamsLGSSMEmissions


class PosteriorGSSMFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$

    """
    marginal_loglik: Union[Scalar, Float[Array, " ntime"]]
    filtered_means: Optional[Float[Array, "ntime state_dim"]] = None
    filtered_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None
    predicted_means: Optional[Float[Array, "ntime state_dim"]] = None
    predicted_covariances: Optional[Float[Array, "ntime state_dim state_dim"]] = None


class PosteriorGSSMSmoothed(NamedTuple):
    r"""Marginals of the Gaussian filtering and smoothing posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    :param smoothed_means: array of smoothed means $\mathbb{E}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_covariances: array of smoothed marginal covariances, $\mathrm{Cov}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_cross_covariances: array of smoothed cross products, $\mathbb{E}[z_t z_{t+1}^T \mid y_{1:T}, u_{1:T}]$

    """
    marginal_loglik: Scalar
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "ntime_minus1 state_dim state_dim"]] = None


# Helper functions

def _get_one_param(x, dim, t):
    """Helper function to get one parameter at time t."""
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x

def _get_params(params, num_timesteps, t):
    """Helper function to get all parameters at time t."""
    assert not callable(params.emissions.cov), "Emission covariance cannot be a callable."

    F = _get_one_param(params.dynamics.weights, 2, t)
    B = _get_one_param(params.dynamics.input_weights, 2, t)
    b = _get_one_param(params.dynamics.bias, 1, t)
    Q = _get_one_param(params.dynamics.cov, 2, t)
    H = _get_one_param(params.emissions.weights, 2, t)
    D = _get_one_param(params.emissions.input_weights, 2, t)
    d = _get_one_param(params.emissions.bias, 1, t)

    if len(params.emissions.cov.shape) == 1:
        R = _get_one_param(params.emissions.cov, 1, t)
    elif len(params.emissions.cov.shape) > 2:
        R = _get_one_param(params.emissions.cov, 2, t)
    elif params.emissions.cov.shape[0] != num_timesteps:
        R = _get_one_param(params.emissions.cov, 2, t)
    elif params.emissions.cov.shape[1] != num_timesteps:
        R = _get_one_param(params.emissions.cov, 1, t)
    else:
        R = _get_one_param(params.emissions.cov, 2, t)
        warnings.warn(
            "Emission covariance has shape (N,N) where N is the number of timesteps. "
            "The covariance will be interpreted as static and non-diagonal. To "
            "specify a dynamic and diagonal covariance, pass it as a 3D array.")

    return F, B, b, Q, H, D, d, R


_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)


def make_lgssm_params(initial_mean: Float[Array, " state_dim"],
                      initial_cov: Float[Array, "state_dim state_dim"],
                      dynamics_weights: Float[Array, "state_dim state_dim"],
                      dynamics_cov: Float[Array, "state_dim state_dim"],
                      emissions_weights: Float[Array, "emission_dim state_dim"],
                      emissions_cov: Float[Array, "emission_dim emission_dim"],
                      dynamics_bias: Optional[Float[Array, " state_dim"]]=None,
                      dynamics_input_weights: Optional[Float[Array, "state_dim input_dim"]]=None,
                      emissions_bias: Optional[Float[Array, " emission_dim"]]=None,
                      emissions_input_weights: Optional[Float[Array, "emission_dim input_dim"]]=None
                      ) -> ParamsLGSSM:
    """Helper function to construct a ParamsLGSSM object from arguments.

    See `ParamsLGSSM`, `ParamsLGSSMInitial`, `ParamsLGSSMDynamics`, and `ParamsLGSSMEmissions` for
    more details on the parameters.
    """
    state_dim = len(initial_mean)
    emission_dim = emissions_cov.shape[-1]
    input_dim = max(dynamics_input_weights.shape[-1] if dynamics_input_weights is not None else 0,
                    emissions_input_weights.shape[-1] if emissions_input_weights is not None else 0)

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=initial_mean,
            cov=initial_cov
        ),
        dynamics=ParamsLGSSMDynamics(
            weights=dynamics_weights,
            bias=_zeros_if_none(dynamics_bias,state_dim),
            input_weights=_zeros_if_none(dynamics_input_weights, (state_dim, input_dim)),
            cov=dynamics_cov
        ),
        emissions=ParamsLGSSMEmissions(
            weights=emissions_weights,
            bias=_zeros_if_none(emissions_bias, emission_dim),
            input_weights=_zeros_if_none(emissions_input_weights, (emission_dim, input_dim)),
            cov=emissions_cov
        )
    )
    return params


def _predict(prior_mean: Float[Array, "state_dim"],
             prior_cov: Float[Array, "state_dim state_dim"],
             dynamics_matrix: Float[Array, "state_dim state_dim"],
             input_weights: Float[Array, "state_dim input_dim"],
             dynamics_bias: Float[Array, "state_dim"],
             dynamics_cov: Float[Array, "state_dim state_dim"],
             inpt: Float[Array, "input_dim"]
             ) -> Tuple[Float[Array, "state_dim"], 
                        Float[Array, "state_dim state_dim"]]:
    r"""Predict next mean and covariance under a linear Gaussian model.

        p(z_{t+1}) = int N(z_t \mid m, S) N(z_{t+1} \mid Fz_t + Bu + b, Q)
                    = N(z_{t+1} \mid Fm + Bu, F S F^T + Q)

    Returns:
        mu_pred (state_dim,): predicted mean.
        Sigma_pred (state_dim,state_dim): predicted covariance.
    """
    mu_pred = dynamics_matrix @ prior_mean + input_weights @ inpt + dynamics_bias
    Sigma_pred = dynamics_matrix @ prior_cov @ dynamics_matrix.T + dynamics_cov
    return mu_pred, Sigma_pred


def _condition_on(prior_mean: Float[Array, "state_dim"],
                  prior_cov: Float[Array, "state_dim state_dim"],
                  emission_matrix: Float[Array, "emission_dim state_dim"],
                  input_weights: Float[Array, "emission_dim input_dim"],
                  emission_bias: Float[Array, "emission_dim"],
                  emission_cov: Union[Float[Array, "emission_dim emission_dim"], Float[Array, "emission_dim"]],
                  inpt: Float[Array, "input_dim"],
                  emission: Float[Array, "emission_dim"]):
    r"""Condition a Gaussian potential on a new linear Gaussian observation
       p(z_t \mid y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t \mid y_{1:t-1}, u_{1:t-1}) p(y_t \mid z_t, u_t)
         = N(z_t \mid m, P) N(y_t \mid H_t z_t + D_t u_t + d_t, R_t)
         = N(z_t \mid mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    if emission_cov.ndim == 2:
        S = emission_cov + emission_matrix @ prior_cov @ emission_matrix.T
        K = psd_solve(S, emission_matrix @ prior_cov).T
    else:
        # Optimization using Woodbury identity with A=R, U=H@chol(P), V=U.T, C=I
        # (see https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
        I = jnp.eye(prior_cov.shape[0])
        U = emission_matrix @ jnp.linalg.cholesky(prior_cov)
        X = U / emission_cov[:, None]
        S_inv = jnp.diag(1.0 / emission_cov) - X @ psd_solve(I + U.T @ X, X.T)
        """
        # Could alternatively use U=H and C=P
        R_inv = jnp.diag(1.0 / R)
        P_inv = psd_solve(P, jnp.eye(P.shape[0]))
        S_inv = R_inv - R_inv @ H @ psd_solve(P_inv + H.T @ R_inv @ H, H.T @ R_inv)
        """
        K = prior_cov @ emission_matrix.T @ S_inv
        S = jnp.diag(emission_cov) + emission_matrix @ prior_cov @ emission_matrix.T

    residual = emission - input_weights @ inpt - emission_bias - emission_matrix @ prior_mean
    mu_cond = prior_mean + K @ residual
    Sigma_cond = prior_cov - K @ S @ K.T
    return mu_cond, symmetrize(Sigma_cond)


def preprocess_params_and_inputs(params: ParamsLGSSM, 
                                 num_timesteps: int, 
                                 inputs: Optional[Float[Array, "num_timesteps input_dim"]]
                                 ) -> Tuple[ParamsLGSSM, 
                                            Float[Array, "num_timesteps input_dim"]]:
    """Preprocess parameters in case some are set to None.
    
    Args:
        params: model parameters
        num_timesteps: number of timesteps
        inputs: optional array of inputs.

    Returns:
        full_params: full parameters with zeros for missing parameters
        inputs: processed inputs (zero if None)
    """

    # Make sure all the required parameters are there
    assert params.initial.mean is not None
    assert params.initial.cov is not None
    assert params.dynamics.weights is not None
    assert params.dynamics.cov is not None
    assert params.emissions.weights is not None
    assert params.emissions.cov is not None

    # Get shapes
    emission_dim, state_dim = params.emissions.weights.shape[-2:]

    # Default the inputs to zero
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))
    input_dim = inputs.shape[-1]

    # Default other parameters to zero
    dynamics_input_weights = _zeros_if_none(params.dynamics.input_weights, (state_dim, input_dim))
    dynamics_bias = _zeros_if_none(params.dynamics.bias, (state_dim,))
    emissions_input_weights = _zeros_if_none(params.emissions.input_weights, (emission_dim, input_dim))
    emissions_bias = _zeros_if_none(params.emissions.bias, (emission_dim,))

    full_params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=params.initial.mean,
            cov=params.initial.cov),
        dynamics=ParamsLGSSMDynamics(
            weights=params.dynamics.weights,
            bias=dynamics_bias,
            input_weights=dynamics_input_weights,
            cov=params.dynamics.cov),
        emissions=ParamsLGSSMEmissions(
            weights=params.emissions.weights,
            bias=emissions_bias,
            input_weights=emissions_input_weights,
            cov=params.emissions.cov)
        )
    return full_params, inputs


def preprocess_args(f):
    """Preprocess the parameter and input arguments in case some are set to None."""
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        """Wrapper function to preprocess arguments."""
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments['params']
        emissions = bound_args.arguments['emissions']
        inputs = bound_args.arguments['inputs']

        num_timesteps = len(emissions)
        full_params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

        return f(full_params, emissions, inputs=inputs)
    return wrapper


def lgssm_joint_sample(params: ParamsLGSSM,
                       key: PRNGKeyT,
                       num_timesteps: int,
                       inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
                       )-> Tuple[Float[Array, "num_timesteps state_dim"],
                                 Float[Array, "num_timesteps emission_dim"]]:
    r"""Sample from the joint distribution to produce state and emission trajectories.

    Args:
        params: model parameters
        key: random number key.
        num_timesteps: number of timesteps.
        inputs: optional array of inputs.

    Returns:
        latent states and emissions sampled from the model.
    """
    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

    def _sample_transition(key, F, B, b, Q, x_tm1, u):
        """Sample from the transition distribution."""
        mean = F @ x_tm1 + B @ u + b
        return MVN(mean, Q).sample(seed=key)

    def _sample_emission(key, H, D, d, R, x, u):
        """Sample from the emission distribution."""
        mean = H @ x + D @ u + d
        R = jnp.diag(R) if R.ndim==1 else R
        return MVN(mean, R).sample(seed=key)

    def _sample_initial(key, params, inputs):
        """Sample from the initial distribution."""
        key1, key2 = jr.split(key)

        initial_state = MVN(params.initial.mean, params.initial.cov).sample(seed=key1)

        H0, D0, d0, R0 = _get_params(params, num_timesteps, 0)[4:]
        u0 = tree_map(lambda x: x[0], inputs)

        initial_emission = _sample_emission(key2, H0, D0, d0, R0, initial_state, u0)
        return initial_state, initial_emission

    def _step(prev_state, args):
        """Sample the next state and emission."""
        key, t, inpt = args
        key1, key2 = jr.split(key, 2)

        # Get parameters and inputs for time index t
        F, B, b, Q, H, D, d, R = _get_params(params, num_timesteps, t)

        # Sample from transition and emission distributions
        state = _sample_transition(key1, F, B, b, Q, prev_state, inpt)
        emission = _sample_emission(key2, H, D, d, R, state, inpt)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)

    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Sample the remaining emissions and states
    next_keys = jr.split(key2, num_timesteps - 1)
    next_times = jnp.arange(1, num_timesteps)
    next_inputs = tree_map(lambda x: x[1:], inputs)
    _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_times, next_inputs))

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


@preprocess_args
def lgssm_filter(params: ParamsLGSSM,
                 emissions:  Float[Array, "ntime emission_dim"],
                 inputs: Optional[Float[Array, "ntime input_dim"]]=None
                 ) -> PosteriorGSSMFiltered:
    r"""Run a Kalman filter to produce the marginal likelihood and filtered state estimates.

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.

    Returns:
        PosteriorGSSMFiltered: filtered posterior object

    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    def _log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y):
        """Compute the log likelihood of an observation under a linear Gaussian model."""
        m = H @ pred_mean + D @ u + d
        if R.ndim==2:
            S = R + H @ pred_cov @ H.T
            return MVN(m, S).log_prob(y)
        else:
            L = H @ jnp.linalg.cholesky(pred_cov)
            return MVNLowRank(m, R, L).log_prob(y)


    def _step(carry, t):
        """Run one step of the Kalman filter."""
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        F, B, b, Q, H, D, d, R = _get_params(params, num_timesteps, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        ll += _log_likelihood(pred_mean, pred_cov, H, D, d, R, u, y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial.mean, params.initial.cov)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


@preprocess_args
def lgssm_smoother(params: ParamsLGSSM,
                   emissions: Float[Array, "ntime emission_dim"],
                   inputs: Optional[Float[Array, "ntime input_dim"]]=None
                   ) -> PosteriorGSSMSmoothed:
    r"""Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        emissions: array of observations.
        inputs: array of inputs.

    Returns:
        PosteriorGSSMSmoothed: smoothed posterior object.

    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Run the smoother backward in time
    def _step(carry, args):
        """Run one step of the Kalman smoother."""
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        F, B, b, Q = _get_params(params, num_timesteps, t)[:4]
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = psd_solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )


def lgssm_posterior_sample(key: PRNGKeyT,
                           params: ParamsLGSSM,
                           emissions:  Float[Array, "num_timesteps emission_dim"],
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
                           jitter: Optional[Scalar]=0.0
                           ) -> Float[Array, "num_timesteps state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples from $p(z_{1:T} \mid y_{1:T}, u_{1:T})$.

    Args:
        key: random number key.
        params: parameters.
        emissions: sequence of observations.
        inputs: optional sequence of inptus.
        jitter: padding to add to the diagonal of the covariance matrix before sampling.

    Returns:
        One sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Sample backward in time
    def _step(carry, args):
        """Run one step of the backward sampling algorithm."""
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        F, B, b, Q = _get_params(params, num_timesteps, t)[:4]
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, F, B, b, Q, u, next_state)
        smoothed_cov = smoothed_cov + jnp.eye(smoothed_cov.shape[-1]) * jitter
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    _, states = lax.scan(
        _step,
        last_state,
        (
            jr.split(key, num_timesteps - 1),
            filtered_means[:-1],
            filtered_covs[:-1],
            jnp.arange(num_timesteps - 1),
        ),
        reverse=True,
    )

    return jnp.vstack([states, last_state])
