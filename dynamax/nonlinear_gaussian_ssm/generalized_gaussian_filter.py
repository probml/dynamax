from jax import numpy as jnp
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from typing import Callable
from itertools import product
from dataclasses import dataclass

from numpy.polynomial.hermite_e import hermegauss
from jax import jacfwd
from jax import vmap
import jax.numpy as jnp
import chex

from jaxtyping import Array, Float, PyTree, Bool, Int, Num
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union,  TypeVar, Generic, Mapping, Callable

from dynamax.nonlinear_gaussian_ssm.nonlinear_gaussian_ssm import PosteriorNLGSSMFiltered, PosteriorNLGSSMSmoothed, ParamsNLGSSM


@chex.dataclass
class ParamsGGSSM(ParamsNLGSSM):
    """Lightweight container for GGSSM parameters. Extends NLGSSM with functions to compute Gaussian moments."""
    gaussian_expectation: Callable
    gaussian_cross_covariance: Callable



@chex.dataclass
class EKFParams(ParamsGGSSM):
    """
    Lightweight container for extended Kalman filter/smoother parameters.
    """
    gaussian_expectation: Callable = lambda f, m, P: f(m)
    gaussian_cross_covariance: Callable = lambda f, g, m, P: jacfwd(f)(m) @ P @ jacfwd(g)(m).T


@dataclass
class SigmaPointParams(ParamsGGSSM):
    """
    Lightweight container for sigma point filter/smoother parameters.
    """
    def _gaussian_expectation(self, f, m, P):
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1)

    def _gaussian_cross_covariance(self, f, g, m, P):
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(g, m, P)
        return jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1)


@dataclass
class UKFParams(SigmaPointParams):
    """
    Lightweight container for unscented Kalman filter/smoother parameters.
    """
    alpha: chex.Scalar = jnp.sqrt(3)
    beta: chex.Scalar = 2
    kappa: chex.Scalar = 1
    compute_weights_and_sigmas: Callable = lambda x, y: (0, 0, 0)
    gaussian_expectation: Callable = None
    gaussian_cross_covariance: Callable = None
    
    def __post_init__(self):
        self.compute_weights_and_sigmas = self._compute_weights_and_sigmas
        self.gaussian_expectation = super()._gaussian_expectation
        self.gaussian_cross_covariance = super()._gaussian_cross_covariance

    def _compute_weights_and_sigmas(self, m, P):
        n = len(self.initial_mean)
        lamb = self.alpha**2 * (n + self.kappa) - n
        # Compute weights
        factor = 1 / (2 * (n + lamb))
        w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
        w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - self.alpha**2 + self.beta)]), jnp.ones(2 * n) * factor))
        # Compute sigmas
        distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
        sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
        sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
        sigmas = jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))
        return w_mean, w_cov, sigmas


@dataclass
class GHKFParams(SigmaPointParams):
    """
    Lightweight container for Gauss-Hermite Kalman filter/smoother parameters.
    """
    order: chex.Scalar = 10
    compute_weights_and_sigmas: Callable = lambda x, y: (0, 0, 0)
    gaussian_expectation: Callable = None
    gaussian_cross_covariance: Callable = None
    
    def __post_init__(self):
        self.compute_weights_and_sigmas = self._compute_weights_and_sigmas
        self.gaussian_expectation = super()._gaussian_expectation
        self.gaussian_cross_covariance = super()._gaussian_cross_covariance

    def _compute_weights_and_sigmas(self, m, P):
        n = len(self.initial_mean)
        samples_1d, weights_1d = hermegauss(self.order)
        weights_1d /= weights_1d.sum()
        weights = jnp.prod(jnp.array(list(product(weights_1d, repeat=n))), axis=1)
        unit_sigmas = jnp.array(list(product(samples_1d, repeat=n)))
        sigmas = m + vmap(jnp.matmul, [None, 0], 0)(jnp.linalg.cholesky(P), unit_sigmas)
        return weights, weights, sigmas
        
# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x


def _predict(m, P, f, Q, u, g_ev, g_cov):
    """Predict next mean and covariance under an additive-noise Gaussian filter

        p(x_{t+1}) = N(x_{t+1} | mu_pred, Sigma_pred)
        where
            mu_pred = gev(f, m, P)
                    = \int f(x_t, u) N(x_t | m, P) dx_t
            Sigma_pred = gev((f - mu_pred)(f - mu_pred)^T, m, P) + Q
                       = \int (f(x_t, u) - mu_pred)(f(x_t, u) - mu_pred)^T
                           N(x_t | m, P)dx_t + Q

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.
        g_ev (Callable): Gaussian expectation value function.
        g_cov (Callable): Gaussian cross covariance function.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
        cross_pred (D_hid,D_hid): cross covariance term.
    """
    dynamics_fn = lambda x: f(x, u)
    identity_fn = lambda x: x
    mu_pred = g_ev(dynamics_fn, m, P)
    Sigma_pred = g_cov(dynamics_fn, dynamics_fn, m, P) + Q
    cross_pred = g_cov(identity_fn, dynamics_fn, m, P)
    return mu_pred, Sigma_pred, cross_pred


def _condition_on(m, P, h, R, u, y, g_ev, g_cov):
    """Condition a Gaussian potential on a new observation
       p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
         = N(x_t | m, P) N(y_t | h(x_t, u), R_t)
         = N(x_t | mu_cond, Sigma_cond)
     where
        mu_cond = m + K*(y - yhat)
        yhat = gev(h, m, P)
        S = gev((h - yhat)(h - yhat)^T, m, P) + R
        C = gev((Identity - m)(h - yhat)^T, m, P)
        K = C * S^{-1}
        Sigma_cond = P - K S K'

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        h (Callable): emission function.
        R (D_obs,D_obs): emission covariance matrix.
        u (D_in,): inputs.
        y (D_obs,): observation.
        g_ev (Callable): Gaussian expectation value function.
        g_cov (Callable): Gaussian cross covariance function.

     Returns:
        log_likelihood (Scalar): prediction log likelihood for observation y
        mu_cond (D_hid,): conditioned mean.
        Sigma_cond (D_hid,D_hid): conditioned covariance.
    """
    emission_fn = lambda x: h(x, u)
    identity_fn = lambda x: x
    yhat = g_ev(emission_fn, m, P)
    S = g_cov(emission_fn, emission_fn, m, P) + R
    log_likelihood = MVN(yhat, S).log_prob(jnp.atleast_1d(y))
    C = g_cov(identity_fn, emission_fn, m, P)
    K = jnp.linalg.solve(S, C.T).T
    mu_cond = m + K @ (y - yhat)
    Sigma_cond = P - K @ S @ K.T
    return log_likelihood, mu_cond, Sigma_cond


def general_gaussian_filter(
    params: ParamsGGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorNLGSSMFiltered:
    num_timesteps = len(emissions)

    # Process dynamics and emission functions to take in control inputs
    f, h = params.dynamics_function, params.emission_function
    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    # Gaussian expectation value function
    g_ev = params.gaussian_expectation
    g_cov = params.gaussian_cross_covariance

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, h, R, u, y, g_ev, g_cov)
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the general Gaussian filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorNLGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def general_gaussian_smoother(
    params: ParamsGGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorNLGSSMSmoothed:
    num_timesteps = len(emissions)

    # Run the general Gaussian filter
    filtered_posterior = general_gaussian_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Process dynamics and emission functions to take in control inputs
    f, h = params.dynamics_function, params.emission_function
    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    # Gaussian expectation value function
    g_ev = params.gaussian_expectation
    g_cov = params.gaussian_cross_covariance

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]

        # Prediction step
        pred_mean, pred_cov, pred_cross = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)
        G = jnp.linalg.solve(pred_cov, pred_cross.T).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - pred_mean)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - pred_cov) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    return PosteriorNLGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )