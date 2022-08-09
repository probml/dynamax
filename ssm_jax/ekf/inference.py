import jax.numpy as jnp
from jax import lax
from jax import jacfwd
from distrax import MultivariateNormalFullCovariance as MVN
from ssm_jax.nlgssm.containers import NLGSSMPosterior
from ssm_jax.lgssm.models import LinearGaussianSSM


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x


def _predict(m, P, f, F, Q, u):
    """Predict next mean and covariance using first-order additive EKF

        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | f(x_t, u), Q)
                    = N(x_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
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
    Sigma_pred = F_x @ P @ F_x.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, h, H, R, u, y, num_iter):
    """Condition a Gaussian potential on a new observation
       p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
         = N(x_t | m, S) N(y_t | h_t(x_t, u_t), R_t)
         = N(x_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = jnp.linalg.solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def extended_kalman_filter(params, emissions, num_iter=1, inputs=None):
    """Run an (iterated) extended Kalman filter to produce the 
    marginal likelihood and filtered state estimates.

    Args:
        params: an NLGSSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        num_iter (int): number of re-linearizations around posterior for update step.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: LGSSMPosterior instance containing,
            marginal_log_lik
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """
    num_timesteps = len(emissions)
    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        H_x = H(pred_mean, u)
        ll += MVN(h(pred_mean, u), H_x @ pred_cov @ H_x.T + R).log_prob(jnp.atleast_1d(y))

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, h, H, R, u, y, num_iter)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, f, F, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return NLGSSMPosterior(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def extended_kalman_smoother(params, emissions, inputs=None):
    """Run an extended Kalman (RTS) smoother.

    Args:
        params: an NLGSSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        nlgssm_posterior: LGSSMPosterior instance containing properties of
            filtered and smoothed posterior distributions.
    """
    num_timesteps = len(emissions)

    # Run the extended Kalman filter
    ekf_posterior = extended_kalman_filter(params, emissions, inputs=inputs)
    ll, filtered_means, filtered_covs, *_ = ekf_posterior.to_tuple()

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        F_x = F(filtered_mean, u)

        # Prediction step
        m_pred = f(filtered_mean, u)
        S_pred = Q + F_x @ filtered_cov @ F_x.T
        G = jnp.linalg.solve(S_pred, F_x @ filtered_cov).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the extended Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    return NLGSSMPosterior(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )

def _posterior_linearize(m, P, f, F, h, H, num_timesteps, inputs):
    def _step(_, t):
        mean, cov = m[t], P[t]
        u = inputs[t]
        F_x, b = F(mean, u), f(mean, u) - F(mean, u) @ mean
        H_x, d = H(mean, u), h(mean, u) - H(mean, u) @ mean
        return None, (F_x, b, H_x, d)
    
    _, (Fs, bs, Hs, ds) = lax.scan(_step, None, jnp.arange(num_timesteps))
    return (Fs, bs, Hs, ds)


def iterated_extended_kalman_smoother(params, emissions, num_iter=1, inputs=None):
    num_timesteps = len(emissions)

    # Run first iteration of eks
    post = extended_kalman_smoother(params, emissions, inputs)
    smoothed_means, smoothed_covs = post.smoothed_means, post.smoothed_covariances

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, _):
        means, covs = carry
        Fs, bs, Hs, ds = _posterior_linearize(means, covs, f, F, h, H, num_timesteps, inputs)

        # Run LGSSM smoother
        lgssm = LinearGaussianSSM(
            initial_mean = params.initial_mean,
            initial_covariance = params.initial_covariance,
            dynamics_matrix = Fs,
            dynamics_input_weights = jnp.zeros((Fs.shape[-2], inputs.shape[-1])), # TODO
            dynamics_bias = bs,
            dynamics_covariance = params.dynamics_covariance,
            emission_matrix = Hs,
            emission_input_weights = jnp.zeros((Hs.shape[-2], inputs.shape[-1])), # TODO
            emission_bias = ds,
            emission_covariance = params.emission_covariance
        )
        lgssm_post = lgssm.smoother(emissions, inputs)

        smoothed_means, smoothed_covs = lgssm_post.smoothed_means, lgssm_post.smoothed_covariances
        return (smoothed_means, smoothed_covs), None
    
    carry = (smoothed_means, smoothed_covs)
    (smoothed_means, smoothed_covs), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return NLGSSMPosterior(
        marginal_loglik=post.marginal_loglik,
        filtered_means=post.filtered_means,
        filtered_covariances=post.filtered_covariances,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )