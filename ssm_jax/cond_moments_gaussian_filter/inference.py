from jax import numpy as jnp
from jax import lax
from distrax import MultivariateNormalFullCovariance as MVN
from ssm_jax.cond_moments_gaussian_filter.containers import CMGFPosterior


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


def _condition_on(m, P, y_cond_mean, y_cond_var, u, y, g_ev, g_cov, num_iter):
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
        y_cond_var (Callable): conditional emission variance function.
        u (D_in,): inputs.
        y (D_obs,): observation.
        g_ev (Callable): Gaussian expectation value function.
        g_cov (Callable): Gaussian cross covariance function.
        num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
        log_likelihood (Scalar): prediction log likelihood for observation y
        mu_cond (D_hid,): conditioned mean.
        Sigma_cond (D_hid,D_hid): conditioned covariance.
    """
    m_Y = lambda x: y_cond_mean(x, u)
    Var_Y = lambda x: y_cond_var(x, u)
    identity_fn = lambda x: x

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = g_ev(m_Y, prior_mean, prior_cov)
        S = g_ev(Var_Y, prior_mean, prior_cov) + g_cov(m_Y, m_Y, prior_mean, prior_cov)
        log_likelihood = MVN(yhat, S).log_prob(jnp.atleast_1d(y))
        C = g_cov(identity_fn, m_Y, prior_mean, prior_cov)
        K = jnp.linalg.solve(S, C.T).T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = prior_cov - K @ S @ K.T
        return (posterior_mean, posterior_cov), log_likelihood

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), lls = lax.scan(_step, carry, jnp.arange(num_iter))
    return lls[0], mu_cond, Sigma_cond


def statistical_linear_regression(mu, Sigma, m, S, C):
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
    A = jnp.linalg.solve(Sigma.T, C).T
    b = m - A @ mu
    Omega = S - A @ Sigma @ A.T
    return A, b, Omega


def conditional_moments_gaussian_filter(params, emissions, num_iter=1, inputs=None):
    """Run an (iterated) conditional moments Gaussian filter to produce the 
    marginal likelihood and filtered state estimates.

    Args:
        params: a CMGFParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        num_iter (int): number of linearizations around prior/posterior for update step.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: CMGFPosterior instance containing,
            marginal_log_lik
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """
    num_timesteps = len(emissions)
    
    # Process dynamics function and conditional emission moments to take in control inputs
    f = params.dynamics_function
    m_Y, Var_Y = params.emission_mean_function, params.emission_var_function
    f, m_Y, Var_Y  = (_process_fn(fn, inputs) for fn in (f, m_Y, Var_Y))
    inputs = _process_input(inputs, num_timesteps)

    # Gaussian expectation value function
    g_ev = params.gaussian_expectation
    g_cov = params.gaussian_cross_covariance

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, m_Y, Var_Y, u, y, g_ev, g_cov, num_iter)
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q, u, g_ev, g_cov)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)
    
    # Run the general linearization filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return CMGFPosterior(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def iterated_conditional_moments_gaussian_filter(params, emissions, num_iter=2, inputs=None):
    """Run an iterated conditional moments Gaussian filter.

    Args:
        params: a CMGFParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        num_iter (int): number of linearizations around smoothed posterior.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: CMGFPosterior instance containing,
            marginal_log_lik
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """
    filtered_posterior = conditional_moments_gaussian_filter(params, emissions, num_iter, inputs)
    return filtered_posterior


def conditional_moments_gaussian_smoother(params, emissions, filtered_posterior=None, inputs=None):
    """Run a conditional moments Gaussian smoother.

    Args:
        params: a CMGFParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        filtered_posterior (GSLRPosterior): filtered posterior to use for smoothing.
            If None, the smoother computes the filtered posterior directly.
        inputs (T,D_in): array of inputs.

    Returns:
        nlgssm_posterior: CMGFPosterior instance containing properties of
            filtered and smoothed posterior distributions.
    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = conditional_moments_gaussian_filter(params, emissions, inputs=inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Process dynamics function to take in control inputs
    f  = _process_fn(params.dynamics_function, inputs)
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
    return CMGFPosterior(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )


def iterated_conditional_moments_gaussian_smoother(params, emissions, num_iter=1, inputs=None):
    """Run an iterated conditional moments Gaussian smoother.

    Args:
        params: an CMGFParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        num_iter (int): number of re-linearizations around smoothed posterior.
        inputs (T,D_in): array of inputs.

    Returns:
        nlgssm_posterior: CMGFPosterior instance containing properties of
            filtered and smoothed posterior distributions.
    """
    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = conditional_moments_gaussian_smoother(params, emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior
