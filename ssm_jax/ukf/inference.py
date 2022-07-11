import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import vmap
from jax import jacfwd
from distrax import MultivariateNormalFullCovariance as MVN
from ssm_jax.nlgssm.containers import NLGSSMPosterior
import chex


@chex.dataclass
class UKFHyperParams:
    """Lightweight container for UKF hyperparameters.
    """

    alpha: chex.Scalar = 1
    beta: chex.Scalar = 1
    kappa: chex.Scalar = 2


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)


def _compute_sigmas(m, n, S, lamb):
    """Compute (2n+1) sigma points used for inputs to  unscented transform.

    Args:
        m (D_hid,): mean.
        n (int): number of state dimensions.
        S (D_hid,D_hid): covariance.
        lamb (Scalar): unscented parameter lambda.

    Returns:
        sigmas (2*D_hid+1,): 2n+1 sigma points.
    """
    assert len(m) == n
    distances = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(S)
    sigma_plus = jnp.array([m + distances[:, i] for i in range(n)])
    sigma_minus = jnp.array([m - distances[:, i] for i in range(n)])
    return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))


def _compute_weights(n, alpha, beta, lamb):
    """Compute weights used to compute predicted mean and covariance (Sarkka 5.77).

    Args:
        n (int): number of state dimensions.
        alpha (float): hyperparameter that determines the spread of sigma points
        beta (float): hyperparameter that incorporates prior information
        lamb (float): lamb = alpha**2 *(n + kappa) - n

    Returns:
        w_mean (2*n+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*n+1,): 2n+1 weights to compute predicted covariance.
    """
    factor = 1 / (2 * (n + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
    w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha ** 2 + beta)]), jnp.ones(2 * n) * factor))
    return w_mean, w_cov


def _predict(m, S, f, Q, lamb, w_mean, w_cov, u):
    """Predict next mean and covariance using additive UKF

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        u (D_in,): inputs.

    Returns:
        m_pred (D_hid,): predicted mean.
        S_pred (D_hid,D_hid): predicted covariance.
    """
    n = len(m)
    # Form sigma points and propagate
    sigmas_pred = _compute_sigmas(m, S, n, lamb)
    u_s = jnp.array([u] * len(sigmas_pred))
    sigmas_pred = vmap(f, (0, 0), 0)(sigmas_pred, us)

    # Compute predicted mean and covariance
    m_pred = jnp.tensordot(w_mean, sigmas_pred, axes=1)
    S_pred = jnp.tensordot(w_cov, _outer(sigmas_pred - m_pred, sigmas_pred - m_pred), axes=1) + Q
    return m_pred, S_pred


def _condition_on(m, S, h, R, lamb, w_mean, w_cov, u, y):
    """Condition a Gaussian potential on a new observation

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        h (Callable): emission function.
        R (D_obs,D_obs): emssion covariance matrix
        lamb (float): lamb = alpha**2 *(n + kappa) - n.
        w_mean (2*D_hid+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*D_hid+1,): 2n+1 weights to compute predicted covariance.
        u (D_in,): inputs.
        y (D_obs,): observation.

    Returns:
        ll (float): log-likelihood of observation
        m_cond (D_hid,): filtered mean.
        S_cond (D_hid,D_hid): filtered covariance.
    """
    n = len(m)
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(m, S, n, lamb)
    u_s = jnp.array([u] * len(sigmas_cond))
    sigmas_cond_prop = vmap(h, (0, 0), 0)(sigmas_cond, u_s)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordor(w_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(w_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1) + R
    pred_cross = jnp.tensordot(w_cov, _outer(sigmas_cond - m, sigmas_cond - m), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(y)

    # Compute filtered mean and covariace
    K = jnp.linalg.solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = m + K @ (y - pred_mean)
    S_cond = S - K @ pred_cov @ K.T
    return ll, m_cond, S_cond


def unscented_kalman_filter(params, emissions, hyperparams, inputs=None):
    num_timesteps = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = alpha ** 2 + (state_dim + kappa) - state_dim
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function

    # If no input, add dummy input to functions
    if inputs is None:
        inputs = jnp.zeros((num_timesteps,))
        process_fn = lambda fn: (lambda x, u: fn(x))
        f, h = (process_fn(fn) for fn in (f, h))

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Get parameters and inputs for time t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on this emission
        log_likelihood, filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, h, R, lamb, w_mean, w_cov, u, y
        )

        # Update the log likelihood
        ll += log_likelihood

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the UKF
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return NLGSSMPosterior(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)
