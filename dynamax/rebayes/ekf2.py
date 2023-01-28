from jax import numpy as jnp
from jax import lax, jacrev, jacfwd, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN

from dynamax.rebayes.base import *

_take_diagonal = lambda x: jnp.diag(x) if len(x.shape) == 2 else x
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

def _full_covariance_condition_on(m, P, y_cond_mean, y_cond_cov, u, y, num_iter):
    """Condition on the emission using a full-covariance EKF.
    Note that this method uses `jnp.linalg.lstsq()` to solve the linear system
    to avoid numerical issues with `jnp.linalg.solve()`.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid, D_hid): Posterior covariance.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        S = R + (H @ prior_cov @ H.T)
        C = prior_cov @ H.T
        K = jnp.linalg.lstsq(S, C.T)[0].T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = prior_cov - K @ S @ K.T
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond

def _fully_decoupled_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter):
    """Condition on the emission using a fully decoupled EKF.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid,): Posterior covariance diagonal elements.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        S = R + (vmap(lambda hh, pp: pp * jnp.outer(hh, hh), (1, 0))(H, prior_cov)).sum(axis=0)
        K = prior_cov[:, None] * jnp.linalg.lstsq(S.T, H)[0].T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = prior_cov - prior_cov * vmap(lambda kk, hh: kk @ hh, (0, 1))(K, H)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond

def _variational_diagonal_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter):
    """Condition on the emission using a variational diagonal EKF.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid,): Posterior covariance diagonal elements.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        K = jnp.linalg.lstsq((R + (prior_cov * H) @ H.T).T, prior_cov * H)
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
        posterior_cov = 1/(1/prior_cov + ((H.T @ R_inv) * H.T).sum(-1))
        posterior_mean = prior_mean + K @(y - yhat)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond

class RebayesEKF(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        method: str
    ):
        self.params = params
        if method == 'fcekf':
            self.update_fn = _full_covariance_condition_on
        elif method == 'vdekf':
            self.update_fn = _variational_diagonal_ekf_condition_on
            self.params.initial_covariance = _take_diagonal(params.initial_covariance)
            self.params.dynamics_covariance = _take_diagonal(params.dynamics_covariance)
        elif method == 'fdekf':
            self.update_fn = _fully_decoupled_ekf_condition_on
            self.params.initial_covariance = _take_diagonal(params.initial_covariance)
            self.params.dynamics_covariance = _take_diagonal(params.dynamics_covariance)
        else:
            raise ValueError('unknown method ', method)

    def update_state(self, bel, u, y):
        m, P = bel.mean, bel.cov # p(z(t) | y(1:t-1))
        mu, Sigma = self.update_fn(m, P, self.params.emission_mean_function, self.params.emission_cov_function, u, y, num_iter=1)
        return Gaussian(mean=mu, cov=Sigma)
