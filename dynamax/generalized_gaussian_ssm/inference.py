"""Inference routines for generalized Gaussian state-space models."""
import jax.numpy as jnp

from itertools import product
from jax import jacfwd, vmap, lax
from jax import lax
from jaxtyping import Array, Float
from numpy.polynomial.hermite_e import hermegauss
from typing import Callable, NamedTuple, Optional, Tuple, Union

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

    def gaussian_expectation(self, 
                             f: Callable, 
                             m: Float[Array, "state_dim"], 
                             P: Float[Array, "state_dim state_dim"]) \
                             -> Float[Array, "output_dim"]:
        r"""Approximate the E[f(x)] where x ~ N(m, P) using quadrature.
        
        Args:
            f (Callable): function to approximate the expectation of.
            m (Array): mean of the Gaussian.
            P (Array): covariance of the Gaussian.
        
        Returns:
            expectation (Array): expectation of f.
        """
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.atleast_1d(jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1))

    def gaussian_cross_covariance(self, 
                                  f: Callable, 
                                  g: Callable, 
                                  m: Float[Array, "state_dim"],
                                  P: Float[Array, "state_dim state_dim"]) \
                                  -> Float[Array, "output_dim_f output_dim_g"]:
        r"""Approximate the Gaussian cross-covariance of two functions f and g,
        
            E[(f(x) - E[f(x)])(g(x) - E[g(x)])^T] where x ~ N(m, P)

        using quadrature.

        Args:
            f (Callable): first function.
            g (Callable): second function.
            m (Array): mean of the Gaussian.
            P (Array): covariance of the Gaussian.

        Returns:
            cross_cov (Array): cross-covariance of f and g.
        """
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean = self.gaussian_expectation(f, m, P)
        g_mean = self.gaussian_expectation(g, m, P)
        return jnp.atleast_2d(jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, 
                                                          vmap(g)(sigmas) - g_mean), 
                                                          axes=1))

    def compute_weights_and_sigmas(self, 
                                   m: Float[Array, "state_dim"],
                                   P: Float[Array, "state_dim state_dim"]) \
                                   -> Tuple[Float[Array, "2*state_dim+1"], 
                                            Float[Array, "2*state_dim+1"], 
                                            Float[Array, "2*state_dim+1 state_dim"]]:
        """Compute weights and sigma points for the UKF."""
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

    def gaussian_expectation(self, 
                             f: Callable, 
                             m: Float[Array, "state_dim"], 
                             P: Float[Array, "state_dim state_dim"]) \
                             -> Float[Array, "output_dim_f"]:
        r"""Approximate the E[f(x)] where x ~ N(m, P) using quadrature.
        
        Args:
            f (Callable): function to approximate the expectation of.
            m (Array): mean of the Gaussian.
            P (Array): covariance of the Gaussian.
        
        Returns:
            expectation (Array): expectation of f.
        """
        w_mean, _, sigmas = self.compute_weights_and_sigmas(m, P)
        return jnp.atleast_1d(jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1))

    def gaussian_cross_covariance(self, 
                                  f: Callable, 
                                  g: Callable, 
                                  m: Float[Array, "state_dim"],
                                  P: Float[Array, "state_dim state_dim"]) \
                                  -> Float[Array, "output_dim_f output_dim_g"]:
        r"""Approximate the Gaussian cross-covariance of two functions f and g,
        
            E[(f(x) - E[f(x)])(g(x) - E[g(x)])^T] where x ~ N(m, P)

        using quadrature.

        Args:
            f (Callable): first function.
            g (Callable): second function.
            m (Array): mean of the Gaussian.
            P (Array): covariance of the Gaussian.

        Returns:
            cross_cov (Array): cross-covariance of f and g.
        """
        _, w_cov, sigmas = self.compute_weights_and_sigmas(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(g, m, P)
        return jnp.atleast_2d(jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1))

    def compute_weights_and_sigmas(self, 
                                   m: Float[Array, "state_dim"],
                                   P: Float[Array, "state_dim state_dim"]) \
                                   -> Tuple[Float[Array, "order**state_dim"],
                                            Float[Array, "order**state_dim"],
                                            Float[Array, "order**state_dim state_dim"]]:
        """Compute weights and sigma points for the GHKF."""
        n = len(m)
        samples_1d, weights_1d = jnp.array(hermegauss(self.order))
        weights_1d /= weights_1d.sum()
        weights = jnp.prod(jnp.array(list(product(weights_1d, repeat=n))), axis=1)
        unit_sigmas = jnp.array(list(product(samples_1d, repeat=n)))
        sigmas = m + vmap(jnp.matmul, [None, 0], 0)(jnp.linalg.cholesky(P), unit_sigmas)
        return weights, weights, sigmas


CMGFIntegrals = Union[EKFIntegrals, UKFIntegrals, GHKFIntegrals]


def _predict(prior_mean: Float[Array, "state_dim"], 
             prior_cov: Float[Array, "state_dim state_dim"],
             dynamics_func: Callable, 
             dynamics_cov: Float[Array, "state_dim state_dim"],
             inpt: Float[Array, "input_dim"],
             gaussian_expec: Callable, 
             gaussian_crosscov: Callable) \
             -> Tuple[Float[Array, "state_dim"],
                      Float[Array, "state_dim state_dim"],
                      Float[Array, "state_dim state_dim"]]:
    r"""Predict next mean and covariance under an additive-noise Gaussian filter

        p(x_{t+1}) = N(x_{t+1} | mu_pred, Sigma_pred)
        where
            mu_pred = gaussian_expec(f, m, P)
                    \approx \int f(x_t, u) N(x_t | m, P) dx_t
            Sigma_pred = gaussian_crosscov(f(x_t, u_t), f(x_t, u_t); m, P) + Q
                       \approx \int (f(x_t, u) - mu_pred)(f(x_t, u) - mu_pred)^T N(x_t | m, P)dx_t + Q
            cross_pred = gaussian_crosscov(x_t, f(x_t, u_t); m, P)
                       \approx \int (x_t - m)(f(x_t, u_t) - mu_pred)^T N(x_t | m, P)dx_t

    Args:
        prior_mean (state_dim,): prior mean.
        prior_cov (state_dim, state_dim): prior covariance.
        dynamics_func (Callable): dynamics function.
        dynamics_cov (state_dim, state_dim): dynamics covariance.
        inpt (D_in,): inputs.
        gaussian_expec (Callable): function to approximate the E[f(x)] where x ~ N(m, P).
        gaussian_crosscov (Callable): 
            function to approximate the E[(f(x) - E[f(x)])(g(x) - E[g(x)])^T] where x ~ N(m, P).
        
    Returns:
        mu_pred (state_dim,): predicted mean.
        Sigma_pred (state_dim, state_dim): predicted covariance.
        cross_pred (state_dim, state_dim): cross covariance term.

    """
    dynamics_fn = lambda x: dynamics_func(x, inpt)
    identity_fn = lambda x: x
    mu_pred = gaussian_expec(dynamics_fn, prior_mean, prior_cov)
    Sigma_pred = gaussian_crosscov(dynamics_fn, dynamics_fn, prior_mean, prior_cov) + dynamics_cov
    cross_pred = gaussian_crosscov(identity_fn, dynamics_fn, prior_mean, prior_cov)
    return mu_pred, Sigma_pred, cross_pred


def _condition_on(prior_mean: Float[Array, "state_dim"],
                  prior_cov: Float[Array, "state_dim state_dim"],
                  emission_mean: Callable, 
                  emission_cov: Callable, 
                  inpt: Float[Array, "input_dim"],
                  emission: Float[Array, "emission_dim"],
                  gaussian_expec: Callable, 
                  gaussian_crosscov: Callable, 
                  num_iter: int, 
                  emission_dist: Callable) \
                  -> Tuple[Float[Array, "emission_dim"],
                           Float[Array, "state_dim"],
                           Float[Array, "state_dim state_dim"]]:
    
    r"""Condition a Gaussian potential on a new observation with arbitrary
       likelihood with given functions for conditional moments and make a
       Gaussian approximation.

       p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
         \approx N(x_t | m, P) ArbitraryDist(y_t | emission_mean(x_t, u_t), emission_cov(x_t, u_t))
         \approx N(x_t | mu_cond, Sigma_cond)
     
     where
        mu_cond = m + K*(y - yhat)
        yhat \approx E[h(x); x \sim N(m, P)]
        S \approx E[(h - yhat)(h - yhat)^T; m, P] + R
        C = gaussian_crosscov((Identity - m)(h - yhat)^T, m, P)
        K = C * S^{-1}
        Sigma_cond = P - K S K'

    Args:
        prior_mean (D_hid,): prior mean.
        prior_cov (D_hid,D_hid): prior covariance.
        emission_mean (Callable): conditional emission mean function.
        emission_cov (Callable): conditional emission covariance function.
        inpt (D_in,): inputs.
        emission (D_obs,): observation.
        gaussian_expec (Callable): Gaussian expectation value function.
        gaussian_crosscov (Callable): Gaussian cross covariance function.
        num_iter (int): number of re-linearizations around posterior for update step.
        emission_dist: the observation distribution p(y | x). Constructed from specified mean and covariance.

     Returns:
        log_likelihood (Scalar): prediction log likelihood for observation y
        mu_cond (D_hid,): conditioned mean.
        Sigma_cond (D_hid,D_hid): conditioned covariance.

    """
    m_Y = lambda x: emission_mean(x, inpt)
    Cov_Y = lambda x: emission_cov(x, inpt)
    identity_fn = lambda x: x

    def _step(carry, _):
        """Iteratively re-linearize around the posterior mean and covariance."""
        prior_mean, prior_cov = carry
        yhat = gaussian_expec(m_Y, prior_mean, prior_cov)
        S = gaussian_expec(Cov_Y, prior_mean, prior_cov) + gaussian_crosscov(m_Y, m_Y, prior_mean, prior_cov)
        log_likelihood = emission_dist(yhat, S).log_prob(jnp.atleast_1d(emission)).sum()
        C = gaussian_crosscov(identity_fn, m_Y, prior_mean, prior_cov)
        K = psd_solve(S, C.T).T
        posterior_mean = prior_mean + K @ (emission - yhat)
        posterior_cov = prior_cov - K @ S @ K.T
        return (posterior_mean, posterior_cov), log_likelihood

    # Iterate re-linearization over posterior mean and covariance
    carry = (prior_mean, prior_cov)
    (mu_cond, Sigma_cond), lls = lax.scan(_step, carry, jnp.arange(num_iter))
    return lls[0], mu_cond, Sigma_cond


def conditional_moments_gaussian_filter(model_params: ParamsGGSSM,
                                        inf_params: CMGFIntegrals,
                                        emissions: Float[Array, "num_timesteps emission_dim"],
                                        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
                                        num_iter: int = 1,
                                        ) -> PosteriorGSSMFiltered:
    """Run an (iterated) conditional moments Gaussian filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        model_params: model parameters.
        inf_params: inference parameters that specify how to compute approximate moments.
        emissions: array of observations.
        inputs: optopnal array of inputs.
        num_iter: optional number of linearizations around prior/posterior for update step (default 1).

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
        """One step of the CMGF"""
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(model_params.dynamics_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        log_likelihood, filtered_mean, filtered_cov = \
            _condition_on(pred_mean, pred_cov, m_Y, Cov_Y, u, y, g_ev, g_cov, num_iter, emission_dist)
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the general linearization filter
    carry = (0.0, model_params.initial_mean, model_params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, 
                                 filtered_means=filtered_means, 
                                 filtered_covariances=filtered_covs)


def conditional_moments_gaussian_smoother(model_params: ParamsGGSSM,
                                          inf_params: CMGFIntegrals,
                                          emissions: Float[Array, "num_timesteps emission_dim"],
                                          filtered_posterior: Optional[PosteriorGSSMFiltered]=None,
                                          inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
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
        """One step of the CMGS"""
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
    _, (smoothed_means, smoothed_covs) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True
    )

    # Concatenate the last smoothed mean and covariance
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(marginal_loglik=ll,
                                 filtered_means=filtered_means,
                                 filtered_covariances=filtered_covs,
                                 smoothed_means=smoothed_means,
                                 smoothed_covariances=smoothed_covs)
