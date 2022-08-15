from typing import Callable
from itertools import product
from dataclasses import dataclass, field

from numpy.polynomial.hermite_e import hermegauss
from jax import jacfwd
from jax import vmap
import jax.numpy as jnp
import chex


@chex.dataclass
class CMGFParams:
    """Lightweight container for CMGF parameters.
    """
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_mean_function: Callable
    emission_var_function: Callable
    gaussian_expectation: Callable
    gaussian_cross_covariance: Callable


@chex.dataclass
class CMGFPosterior:
    """Simple wrapper for properties of an CMGF posterior distribution.

    Attributes:
            marginal_loglik: marginal log likelihood of the data
            filtered_means: (T,D_hid) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_covariances: (T,D_hid,D_hid) array,
                Cov[x_t | y_{1:t}, u_{1:t}].
            smoothed_means: (T,D_hid) array,
                E[x_t | y_{1:T}, u_{1:T}].
            smoothed_covs: (T,D_hid,D_hid) array of smoothed marginal covariances,
                Cov[x_t | y_{1:T}, u_{1:T}].
    """
    marginal_loglik: chex.Scalar = None
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None


@chex.dataclass
class EKFParams(CMGFParams):
    """
    Lightweight container for extended Kalman filter/smoother parameters.
    """
    gaussian_expectation: Callable = lambda f, m, P: f(m)
    gaussian_cross_covariance: Callable = lambda f, g, m, P: jacfwd(f)(m) @ P @ jacfwd(g)(m).T


@dataclass
class SigmaPointParams(CMGFParams):
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