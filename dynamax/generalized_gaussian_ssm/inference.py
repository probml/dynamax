from itertools import product
from numpy.polynomial.hermite_e import hermegauss
from jax import jacfwd, vmap, lax
import jax.numpy as jnp
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Callable

from dynamax.utils.utils import psd_solve
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_jacfwd_2d = lambda f, x: jnp.atleast_2d(jacfwd(f)(x))



class EKFIntegrals(NamedTuple):
    """ Lightweight container for EKF Gaussian integrals."""
    gaussian_expectation: Callable = lambda f, m, P: jnp.atleast_1d(f(m))
    gaussian_cross_covariance: Callable = lambda f, g, m, P: _jacfwd_2d(f, m) @ P @ _jacfwd_2d(g, m).T


class UKFIntegrals(NamedTuple):
    """Lightweight container for UKF Gaussian integrals."""
    alpha: float = jnp.sqrt(3)
    beta: float = 2.0
    kappa: float = 1.0

    def gaussian_expectation(self, f, m, P):
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.atleast_1d(jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1))

    def gaussian_cross_covariance(self, f, g, m, P):
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(g, m, P)
        return jnp.atleast_2d(jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1))

    def compute_weights_and_sigmas(self, m, P):
        n = len(m)
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


class GHKFIntegrals(NamedTuple):
    """Lightweight container for GHKF Gaussian integrals."""
    order: int = 10

    def gaussian_expectation(self, f, m, P):
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.atleast_1d(jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1))

    def gaussian_cross_covariance(self, f, g, m, P):
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(g, m, P)
        return jnp.atleast_2d(jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1))

    def compute_weights_and_sigmas(self, m, P):
        n = len(m)
        samples_1d, weights_1d = jnp.array(hermegauss(self.order))
        weights_1d /= weights_1d.sum()
        weights = jnp.prod(jnp.array(list(product(weights_1d, repeat=n))), axis=1)
        unit_sigmas = jnp.array(list(product(samples_1d, repeat=n)))
        sigmas = m + vmap(jnp.matmul, [None, 0], 0)(jnp.linalg.cholesky(P), unit_sigmas)
        return weights, weights, sigmas


CMGFIntegrals = Union[EKFIntegrals, UKFIntegrals, GHKFIntegrals]


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


def _condition_on(m, P, y_cond_mean, y_cond_cov, u, y, g_ev, g_cov, num_iter, emission_dist):
    """Condition a Gaussian potential on a new observation with arbitrary
       likelihood with given functions for conditional moments and make a
       Gaussian approximation.
       p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
         = N(x_t | m, P) ArbitraryDist(y_t |y_cond_mean(x_t), y_cond_cov(x_t))
         \approx N(x_t | mu_cond, Sigma_cond)
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
        y_cond_mean (Callable): conditional emission mean function.
        y_cond_cov (Callable): conditional emission covariance function.
        u (D_in,): inputs.
        y (D_obs,): observation.
        g_ev (Callable): Gaussian expectation value function.
        g_cov (Callable): Gaussian cross covariance function.
        num_iter (int): number of re-linearizations around posterior for update step.
        emission_dist: the observation pdf q. Constructed from specified mean and covariance.

     Returns:
        log_likelihood (Scalar): prediction log likelihood for observation y
        mu_cond (D_hid,): conditioned mean.
        Sigma_cond (D_hid,D_hid): conditioned covariance.

    """
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)
    identity_fn = lambda x: x

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = g_ev(m_Y, prior_mean, prior_cov)
        S = g_ev(Cov_Y, prior_mean, prior_cov) + g_cov(m_Y, m_Y, prior_mean, prior_cov)
        log_likelihood = emission_dist(yhat, S).log_prob(jnp.atleast_1d(y)).sum()
        C = g_cov(identity_fn, m_Y, prior_mean, prior_cov)
        K = psd_solve(S, C.T).T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = prior_cov - K @ S @ K.T
        return (posterior_mean, posterior_cov), log_likelihood

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), lls = lax.scan(_step, carry, jnp.arange(num_iter))
    return lls[0], mu_cond, Sigma_cond


def _statistical_linear_regression(mu, Sigma, m, S, C):
    """Return moment-matching affine coefficients and approximation noise variance
    given joint moments.

        g(x) \approx Ax + b + e where e ~ N(0, Omega)
        p(x) = N(x | mu, Sigma)
        m = E[g(x)]
        S = Var[g(x)]
        C = Cov[x, g(x)]

    Args:
        mu (D_hid): prior mean.
        Sigma (D_hid, D_hid): prior covariance.
        m (D_obs): E[g(x)].
        S (D_obs, D_obs): Var[g(x)]
        C (D_hid, D_obs): Cov[x, g(x)]

    Returns:
        A (D_obs, D_hid): _description_
        b (D_obs):
        Omega (D_obs, D_obs):
    """
    A = psd_solve(Sigma.T, C).T
    b = m - A @ mu
    Omega = S - A @ Sigma @ A.T
    return A, b, Omega


def conditional_moments_gaussian_filter(
    model_params: ParamsGGSSM,
    inf_params: CMGFIntegrals,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMFiltered:
    """Run an (iterated) conditional moments Gaussian filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        model_params: model parameters.
        inf_params: inference parameters that specify how to compute moments.
        emissions: array of observations.
        num_iter: optional number of linearizations around prior/posterior for update step (default 1).
        inputs: optopnal array of inputs.

    Returns:
        filtered_posterior: posterior object.

    """
    num_timesteps = len(emissions)

    # Process dynamics function and conditional emission moments to take in control inputs
    f = model_params.dynamics_function
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    f, m_Y, Cov_Y  = (_process_fn(fn, inputs) for fn in (f, m_Y, Cov_Y))
    inputs = _process_input(inputs, num_timesteps)

    # Gaussian expectation value function
    g_ev = inf_params.gaussian_expectation
    g_cov = inf_params.gaussian_cross_covariance

    # Emission distribution
    emission_dist = model_params.emission_dist

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(model_params.dynamics_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, m_Y, Cov_Y, u, y, g_ev, g_cov, num_iter, emission_dist)
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the general linearization filter
    carry = (0.0, model_params.initial_mean, model_params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def iterated_conditional_moments_gaussian_filter(
    model_params: ParamsGGSSM,
    inf_params: CMGFIntegrals,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMFiltered:
    """Run an iterated conditional moments Gaussian filter.

    Args:
        model_params: model parameters.
        inf_params: inference parameters that specify how to compute moments.
        emissions: array of observations.
        num_iter: optional number of linearizations around prior/posterior for update step (default 1).
        inputs: optional array of inputs.

    Returns:
        filtered_posterior: posterior object.
    """
    filtered_posterior = conditional_moments_gaussian_filter(model_params, inf_params, emissions, num_iter, inputs)
    return filtered_posterior


def conditional_moments_gaussian_smoother(
    model_params: ParamsGGSSM,
    inf_params: CMGFIntegrals,
    emissions: Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered]=None,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMSmoothed:
    """Run a conditional moments Gaussian smoother.

    Args:
        model_params: model parameters.
        inf_params: inference parameters that specify how to compute moments.
        emissions: array of observations.
        num_iter: optional number of linearizations around prior/posterior for update step (default 1).
        inputs: optopnal array of inputs.

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = conditional_moments_gaussian_filter(model_params, inf_params, emissions, inputs=inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Process dynamics function to take in control inputs
    f  = _process_fn(model_params.dynamics_function, inputs)
    inputs = _process_input(inputs, num_timesteps)

    # Gaussian expectation value function
    g_ev = inf_params.gaussian_expectation
    g_cov = inf_params.gaussian_cross_covariance

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = _get_params(model_params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Prediction step
        pred_mean, pred_cov, pred_cross = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)
        G = psd_solve(pred_cov, pred_cross.T).T

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
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )


def iterated_conditional_moments_gaussian_smoother(
    model_params: ParamsGGSSM,
    inf_params: CMGFIntegrals,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMSmoothed:
    """Run an iterated conditional moments Gaussian smoother.

    Args:
        model_params: model parameters.
        inf_params: inference parameters that specify how to compute moments.
        emissions: array of observations.
        num_iter: optional number of linearizations around prior/posterior for update step (default 1).
        inputs: optopnal array of inputs.

    Returns:
        post: posterior object.

    """
    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = conditional_moments_gaussian_smoother(model_params, inf_params, emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior
