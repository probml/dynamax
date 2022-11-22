import jax.numpy as jnp
import jax.random as jr
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from functools import wraps
import inspect

from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union

from dynamax.utils.utils import psd_solve
from dynamax.parameters import ParameterProperties
from dynamax.types import PRNGKey, Scalar

class ParamsLGSSMInitial(NamedTuple):
    r"""Parameters of the initial distribution

    $$p(x_1) = \mathcal{N}(x_1 \mid \mu_1, Q_1)$$

    The tuple doubles as a container for the ParameterProperties.

    :param mean: $\mu_1$
    :param cov: $Q_1$

    """
    mean: Union[Float[Array, "state_dim"], ParameterProperties]
    # unconstrained parameters are stored as a vector.
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]


class ParamsLGSSMDynamics(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(x_{t+1} \mid x_t, u_t) = \mathcal{N}(x_{t+1} \mid F x_t + B u_t + b, Q)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: dynamics weights $F$
    :param bias: dynamics bias $b$
    :param input_weights: dynamics input weights $B$
    :param cov: dynamics covariance $Q$

    """
    weights: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], ParameterProperties]
    bias: Union[Float[Array, "state_dim"], Float[Array, "ntime state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "state_dim input_dim"], Float[Array, "ntime state_dim input_dim"], ParameterProperties]
    cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "ntime state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]


class ParamsLGSSMEmissions(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(y_t \mid x_t, u_t) = \mathcal{N}(y_t \mid H x_t + D u_t + d, R)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: emission weights $H$
    :param bias: emission bias $d$
    :param input_weights: emission input weights $D$
    :param cov: emission covariance $R$

    """
    weights: Union[Float[Array, "emission_dim state_dim"], Float[Array, "ntime emission_dim state_dim"], ParameterProperties]
    bias: Union[Float[Array, "emission_dim"], Float[Array, "ntime emission_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "emission_dim input_dim"], Float[Array, "ntime emission_dim input_dim"], ParameterProperties]
    cov: Union[Float[Array, "emission_dim emission_dim"], Float[Array, "ntime emission_dim emission_dim"], Float[Array, "emission_dim_triu"], ParameterProperties]



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
    :param filtered_means: array of filtered means $\mathbb{E}[x_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[x_t \mid y_{1:t}, u_{1:t}]$

    """
    marginal_loglik: Scalar
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]


class PosteriorGSSMSmoothed(NamedTuple):
    r"""Marginals of the Gaussian filtering and smoothing posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[x_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[x_t \mid y_{1:t}, u_{1:t}]$
    :param smoothed_means: array of smoothed means $\mathbb{E}[x_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_covariances: array of smoothed marginal covariances, $\mathrm{Cov}[x_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_cross_covariances: array of smoothed cross products, $\mathbb{E}[x_t x_{t+1}^T \mid y_{1:T}, u_{1:T}]$

    """
    marginal_loglik: Scalar
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances: Float[Array, "ntime state_dim state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "ntime_minus1 state_dim state_dim"]] = None


# Helper functions
# _get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
def _get_params(x, dim, t):
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)


def _predict(m, S, F, B, b, Q, u):
    r"""Predict next mean and covariance under a linear Gaussian model.

        p(x_{t+1}) = int N(x_t \mid m, S) N(x_{t+1} \mid Fx_t + Bu + b, Q)
                    = N(x_{t+1} \mid Fm + Bu, F S F^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        F (D_hid,D_hid): dynamics matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        b (D_hid,): dynamics bias.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    mu_pred = F @ m + B @ u + b
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, H, D, d, R, u, y):
    r"""Condition a Gaussian potential on a new linear Gaussian observation
       p(x_t \mid y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t \mid y_{1:t-1}, u_{1:t-1}) p(y_t \mid x_t, u_t)
         = N(x_t \mid m, P) N(y_t \mid H_t x_t + D_t u_t + d_t, R_t)
         = N(x_t \mid mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    S = R + H @ P @ H.T
    K = psd_solve(S, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - D @ u - d - H @ m)
    return mu_cond, Sigma_cond


def preprocess_args(f):
    """Preprocess the parameters and inputs in case some are set to None."""
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments['params']
        emissions = bound_args.arguments['emissions']
        inputs = bound_args.arguments['inputs']

        # Make sure all the required parameters are there
        assert params.initial.mean is not None
        assert params.initial.cov is not None
        assert params.dynamics.weights is not None
        assert params.dynamics.cov is not None
        assert params.emissions.weights is not None
        assert params.emissions.cov is not None

        # Get shapes
        emission_dim, state_dim = params.emissions.weights.shape[-2:]
        num_timesteps = len(emissions)

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
        return f(full_params, emissions, inputs=inputs)
    return wrapper


@preprocess_args
def lgssm_filter(
    params: ParamsLGSSM,
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

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics.weights, 2, t)
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        H = _get_params(params.emissions.weights, 2, t)
        D = _get_params(params.emissions.input_weights, 2, t)
        d = _get_params(params.emissions.bias, 1, t)
        R = _get_params(params.emissions.cov, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        ll += MVN(H @ pred_mean + D @ u + d, H @ pred_cov @ H.T + R).log_prob(y)

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
def lgssm_smoother(
    params: ParamsLGSSM,
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
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics.weights, 2, t)
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = psd_solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )


def lgssm_posterior_sample(
    key: PRNGKey,
    params: ParamsLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples from $p(x_{1:T} \mid y_{1:T}, u_{1:T})$.

    Args:
        key: random number key.
        params: parameters.
        emissions: sequence of observations.
        inputs: optional sequence of inptus.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $x_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics.weights, 2, t)
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, F, B, b, Q, u, next_state)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    args = (
        jr.split(key, num_timesteps - 1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
        jnp.arange(num_timesteps - 2, -1, -1),
    )
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return states
