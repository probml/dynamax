from jax import numpy as jnp
from jax import lax, jacrev, vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN
import chex
from typing import Callable

from dynamax.generalized_gaussian_ssm.dekf.utils import *
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered


_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

@chex.dataclass
class DEKFParams:
    """
    Lightweight container for diagonal EKF Parameters.
    """
    initial_mean: chex.Array
    initial_cov_diag: chex.Array
    dynamics_cov_diag: chex.Array
    emission_mean_function: Callable
    emission_cov_function: Callable


#### Full-covariance
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


#### Fully decoupled
def _stationary_dynamics_diagonal_predict(m, P_diag, Q_diag):
    """Predict the next state using a stationary dynamics model with diagonal covariance matrices.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        Q_diag (D_hid,): Diagonal elements of dynamics covariance.

    Returns:
        mu_pred (D_hid,): Predicted mean.
        Sigma_pred (D_hid,): Predicted covariance diagonal elements.
    """
    mu_pred = m
    Sigma_pred = P_diag + Q_diag
    return mu_pred, Sigma_pred


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

def stationary_dynamics_fully_decoupled_conditional_moments_gaussian_filter(model_params, emissions, num_iter=1, inputs=None):
    """Run a fully decoupled EKF on a stationary dynamics model.

    Args:
        model_params (DEKFParams): Model parameters.
        emissions (T, D_hid): Sequence of emissions.
        num_iter (int, optional): Number of linearizations around posterior for update step.
        inputs (T, D_in, optional): Array of inputs.

    Returns:
        filtered_posterior: GSSMPosterior instance containing,
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """    
    num_timesteps = len(emissions)

    # Process conditional emission moments to take in control inputs
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    m_Y, Cov_Y  = (_process_fn(fn, inputs) for fn in (m_Y, Cov_Y))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        pred_mean, pred_cov_diag = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(model_params.dynamics_cov_diag, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        filtered_mean, filtered_cov_diag = _fully_decoupled_ekf_condition_on(pred_mean, pred_cov_diag, m_Y, Cov_Y, u, y, num_iter)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (model_params.initial_mean, model_params.initial_cov_diag)
    _, (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=None, filtered_means=filtered_means, filtered_covariances=filtered_covs)


### Variational diagonal
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
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
        posterior_cov = 1/(1/prior_cov + ((H.T @ R_inv) * H.T).sum(-1))
        posterior_mean = prior_mean + (posterior_cov * H).T @ R_inv @ (y - yhat)
        # posterior_mean = prior_mean + jnp.diag(posterior_cov) @ H.T @ R_inv @ (y - yhat)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond

def stationary_dynamics_variational_diagonal_extended_kalman_filter(model_params, emissions, num_iter=1, inputs=None):
    """Run a variational diagonal EKF on a stationary dynamics model.

    Args:
        model_params (DEKFParams): Model parameters.
        emissions (T, D_hid): Sequence of emissions.
        num_iter (int, optional): Number of linearizations around posterior for update step.
        inputs (T, D_in, optional): Array of inputs.

    Returns:
        filtered_posterior: GSSMPosterior instance containing,
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """    
    num_timesteps = len(emissions)

    # Process conditional emission moments to take in control inputs
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    m_Y, Cov_Y  = (_process_fn(fn, inputs) for fn in (m_Y, Cov_Y))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        pred_mean, pred_cov_diag = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(model_params.dynamics_cov_diag, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        filtered_mean, filtered_cov_diag = _variational_diagonal_ekf_condition_on(pred_mean, pred_cov_diag, m_Y, Cov_Y, u, y, num_iter)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (model_params.initial_mean, model_params.initial_cov_diag)
    _, (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=None, filtered_means=filtered_means, filtered_covariances=filtered_covs)