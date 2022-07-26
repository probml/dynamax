from typing import Callable

from jax import jacfwd
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