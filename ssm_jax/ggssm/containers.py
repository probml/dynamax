from typing import Callable

from jax import jacfwd
from jax import vmap
import jax.numpy as jnp
import chex


@chex.dataclass
class GGSSMParams:
    """Lightweight container for GGSSM parameters.
    The functions below can be called with an instance of this class.
    However, they can also accept a ssm.ggssm.models.GeneralGaussianSSM instance,
    if you prefer a more object-oriented approach.
    """
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_function: Callable
    dynamics_covariance: chex.Array
    emission_function: Callable
    emission_covariance: chex.Array
    gaussian_expectation: Callable
    gaussian_cross_covariance: Callable


@chex.dataclass
class GGSSMPosterior:
    """Simple wrapper for properties of an GGSSM posterior distribution.

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
class EKFParams(GGSSMParams):
    """
    Lightweight container for extended Kalman filter/smoother parameters.
    """
    gaussian_expectation: Callable = lambda f, m, P: f(m)
    gaussian_cross_covariance: Callable = lambda f, g, m, P: jacfwd(f)(m) @ P @ jacfwd(g)(m).T


@chex.dataclass
class UKFParams(GGSSMParams):
    """
    Lightweight container for unscented Kalman filter/smoother parameters.
    """
    alpha: chex.Scalar = jnp.sqrt(3)
    beta: chex.Scalar = 2
    kappa: chex.Scalar = 1

    # Helper functions
    def _compute_sigmas(self, m, P, n, lamb):
        """Compute (2n+1) sigma points used for inputs to  unscented transform.
        """
        distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
        sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
        sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
        return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))
    
    def _compute_weights(self, n, alpha, beta, lamb):
        """Compute weights used to compute predicted mean and covariance (Sarkka 5.77).
        """
        factor = 1 / (2 * (n + lamb))
        w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
        w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * n) * factor))
        return w_mean, w_cov
    
    def _compute_sigmas_and_weights(self, m, P):
        n = len(self.initial_mean)
        lamb = self.alpha**2 * (n + self.kappa) - n
        w_mean, w_cov = self._compute_weights(n, self.alpha, self.beta, lamb)
        sigmas = self._compute_sigmas(m, P, n, lamb)
        return w_mean, w_cov, sigmas
    
    def gaussian_expectation(self, f, m, P):
        w_mean, _, sigmas = self._compute_sigmas_and_weights(m, P)
        return jnp.tensordot(w_mean, vmap(f)(sigmas), axes=1)

    def gaussian_cross_covariance(self, f, g, m, P):
        w_mean, w_cov, sigmas = self._compute_sigmas_and_weights(m, P)
        _outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
        f_mean, g_mean = self.gaussian_expectation(f, m, P), self.gaussian_expectation(g, m, P)
        return jnp.tensordot(w_cov, _outer(vmap(f)(sigmas) - f_mean, vmap(g)(sigmas) - g_mean), axes=1)