import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd
from distrax import MultivariateNormalFullCovariance as MVN
from ssm_jax.nlgssm.models import NLGSSMParams, NLGSSMPosterior


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim+1 else x

def _predict(m, S, f, F, Q, u):
    """Predict next mean and covariance using first-order additive EKF

        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | f(x_t, u), Q)
                    = N(x_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    F_x = F(m, u)
    mu_pred = f(m, u)
    Sigma_pred = F_x @ S @ F_x.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, S, h, H, R, u, y):
    """Condition a Gaussian potential on a new observation
      p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
        propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
        = N(x_t | m, S) N(y_t | h_t(x_t, u_t), R_t)
        = N(x_t | mm, SS)
    where
        mm = m + K*(y - yhat) = mu_cond
        yhat = h(m, u)
        K = S * H(m, u)' * (R + H(m, u) * S * H(m, u)')^{-1}
        L = I - K*H(m, u)
        SS = L * S * L' + K * R * K' = Sigma_cond
    **Note! This can be done more efficiently when R is diagonal.**

   Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        h (Callable): emission function.
        H (Callable): Jacobian of emission function.
        R (D_obs,D_obs): emission covariance matrix.
        u (D_in,): inputs.
        y (D_obs,): observation.

    Returns:
        mu_cond (D_hid,): filtered mean.
        Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    H_x = H(m, u)
    K = jnp.linalg.solve(R + H_x @ S @ H_x.T, H_x @ S).T
    ImKH = jnp.eye(m.shape[-1]) - K @ H_x
    Sigma_cond = ImKH @ S @ ImKH.T + K @ R @ K.T
    mu_cond = m + K @ (y - h(m, u))
    return mu_cond, Sigma_cond


def extended_kalman_filter(params, emissions, inputs=None):
    """Run an extended Kalman filter to produce the marginal likelihood and 
    filtered state estimates.

    Args:
        params: an ESSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: ESSMPosterior instance containing,
            marginal_log_lik 
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """
    num_timesteps = len(emissions)
    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)
    # If no input, add dummy input to functions
    if inputs is None:
        inputs = jnp.zeros((num_timesteps,))
        process_fn = lambda fn: (lambda x, u: fn(x))
        f, h, F, H = (process_fn(fn) for fn in (f, h, F, H))

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        H_x = H(pred_mean, u)
        ll += MVN(h(pred_mean, u), H_x @ pred_cov @ H_x.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, h, H, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(
            filtered_mean, filtered_cov, f, F, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the extended Kalman filter
    carry = (0., params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return NLGSSMPosterior(marginal_loglik=ll,
                           filtered_means=filtered_means,
                           filtered_covariances=filtered_covs)