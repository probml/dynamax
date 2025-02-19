"""
Unscented Kalman Filter and Smoother for Nonlinear Gaussian State Space Models.
"""
import jax.numpy as jnp
from jax import lax
from jax import vmap
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import Callable, NamedTuple, Optional, List, Tuple

from dynamax.utils.utils import psd_solve
from dynamax.nonlinear_gaussian_ssm.models import  ParamsNLGSSM
from dynamax.linear_gaussian_ssm.models import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

class UKFHyperParams(NamedTuple):
    """Lightweight container for UKF hyperparameters.

    Default values taken from https://github.com/sbitzer/UKF-exposed
    """
    alpha: float = jnp.sqrt(3)
    beta: int = 2
    kappa: int = 1


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_outer = vmap(lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y), 0, 0)
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_compute_lambda = lambda x, y, z: x**2 * (y + z) - z


def _compute_sigmas(mean: Float[Array, "state_dim"], 
                    cov: Float[Array, "state_dim state_dim"], 
                    dim: int, 
                    scale: float) \
                    -> Float[Array, "2*state_dim+1"]:
    """Compute (2n+1) sigma points used for inputs to  unscented transform.

    Args:
        mean (dim,): mean.
        cov (dim, dim): covariance.
        dim (int): number of state dimensions.
        lamb (Scalar): unscented parameter lambda.

    Returns:
        sigmas (2*dim+1,): 2dim+1 sigma points.
    """
    distances = jnp.sqrt(dim + scale) * jnp.linalg.cholesky(cov)
    sigma_plus = jnp.array([mean + distances[:, i] for i in range(dim)])
    sigma_minus = jnp.array([mean - distances[:, i] for i in range(dim)])
    return jnp.concatenate((jnp.array([mean]), sigma_plus, sigma_minus))


def _compute_weights(dim: int, 
                     alpha: float, 
                     beta: float, 
                     lamb: float) \
                     -> Tuple[Float[Array, "2*state_dim+1"], 
                              Float[Array, "2*state_dim+1"]]:
    """Compute weights used to compute predicted mean and covariance (Sarkka 5.77).

    Args:
        dim (int): number of state dimensions.
        alpha (float): hyperparameter that determines the spread of sigma points
        beta (float): hyperparameter that incorporates prior information
        lamb (float): lamb = alpha**2 *(n + kappa) - n

    Returns:
        w_mean (2*n+1,): 2n+1 weights to compute predicted mean.
        w_cov (2*n+1,): 2n+1 weights to compute predicted covariance.
    """
    factor = 1 / (2 * (dim + lamb))
    w_mean = jnp.concatenate((jnp.array([lamb / (dim + lamb)]), jnp.ones(2 * dim) * factor))
    w_cov = jnp.concatenate((jnp.array([lamb / (dim + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * dim) * factor))
    return w_mean, w_cov


def _predict(prior_mean: Float[Array, "state_dim"], 
             prior_cov: Float[Array, "state_dim state_dim"],
             dynamics_func: Callable, 
             dynamics_cov: Float[Array, "state_dim state_dim"],
             lamb: float, 
             weights_mean: Float[Array, "2*state_dim+1"],
             weights_cov: Float[Array, "2*state_dim+1"],
             inpt: Float[Array, "input_dim"]) \
             -> Tuple[Float[Array, "state_dim"],
                      Float[Array, "state_dim state_dim"],
                      Float[Array, "state_dim state_dim"]]:
    """Predict next mean and covariance using additive UKF

    Args:
        prior_mean: prior mean.
        prior_cov: prior covariance.
        dynamics_func: dynamics function.
        dynamics_cov: dynamics covariance matrix.
        lamb: lamb = alpha**2 *(n + kappa) - n.
        weights_mean: 2n+1 weights to compute predicted mean.
        weights_cov: 2n+1 weights to compute predicted covariance.
        inpt: inputs.

    Returns:
        m_pred: predicted mean.
        P_pred: predicted covariance.
        P_cross: predicted cross-covariance.
    """
    n = len(prior_mean)
    # Form sigma points and propagate
    sigmas_pred = _compute_sigmas(prior_mean, prior_cov, n, lamb)
    u_s = jnp.array([inpt] * len(sigmas_pred))
    sigmas_pred_prop = vmap(dynamics_func, (0, 0), 0)(sigmas_pred, u_s)

    # Compute predicted mean and covariance
    m_pred = jnp.tensordot(weights_mean, sigmas_pred_prop, axes=1)
    P_pred = jnp.tensordot(weights_cov, 
                           _outer(sigmas_pred_prop - m_pred, 
                                  sigmas_pred_prop - m_pred), axes=1) \
                                    + dynamics_cov
    P_cross = jnp.tensordot(weights_cov,
                            _outer(sigmas_pred - prior_mean, 
                                   sigmas_pred_prop - m_pred), axes=1)
    return m_pred, P_pred, P_cross


def _condition_on(prior_mean: Float[Array, "state_dim"],
                  prior_cov: Float[Array, "state_dim state_dim"],
                  emission_func: Callable, 
                  emission_cov: Float[Array, "emission_dim emission_dim"],
                  lamb: float, 
                  weights_mean: Float[Array, "2*state_dim+1"],
                  weights_cov: Float[Array, "2*state_dim+1"],
                  inpt: Float[Array, "input_dim"],
                  emission: Float[Array, "emission_dim"]) \
                  -> Tuple[float, 
                           Float[Array, "state_dim"],
                           Float[Array, "state_dim state_dim"]]:
    """Condition a Gaussian potential on a new observation

    Returns:
        ll (float): log-likelihood of observation
        m_cond (D_hid,): filtered mean.
        P_cond (D_hid,D_hid): filtered covariance.

    """
    n = len(prior_mean)
    # Form sigma points and propagate
    sigmas_cond = _compute_sigmas(prior_mean, prior_cov, n, lamb)
    u_s = jnp.array([inpt] * len(sigmas_cond))
    sigmas_cond_prop = vmap(emission_func, (0, 0), 0)(sigmas_cond, u_s)

    # Compute parameters needed to filter
    pred_mean = jnp.tensordot(weights_mean, sigmas_cond_prop, axes=1)
    pred_cov = jnp.tensordot(weights_cov, _outer(sigmas_cond_prop - pred_mean, sigmas_cond_prop - pred_mean), axes=1) + emission_cov
    pred_cross = jnp.tensordot(weights_cov, _outer(sigmas_cond - prior_mean, sigmas_cond_prop - pred_mean), axes=1)

    # Compute log-likelihood of observation
    ll = MVN(pred_mean, pred_cov).log_prob(emission)

    # Compute filtered mean and covariace
    K = psd_solve(pred_cov, pred_cross.T).T  # Filter gain
    m_cond = prior_mean + K @ (emission - pred_mean)
    P_cond = prior_cov - K @ pred_cov @ K.T
    return ll, m_cond, P_cond


def unscented_kalman_filter(params: ParamsNLGSSM,
                            emissions: Float[Array, "ntime emission_dim"],
                            hyperparams: UKFHyperParams,
                            inputs: Optional[Float[Array, "ntime input_dim"]]=None,
                            output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"]) \
                            -> PosteriorGSSMFiltered:
    """Run a unscented Kalman filter to produce the marginal likelihood and
    filtered state estimates.

    Args:
        params: model parameters.
        emissions: array of observations.
        hyperparams: hyper-parameters.
        inputs: optional array of inputs.

    Returns:
        filtered_posterior: posterior object.

    """
    num_timesteps = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        """One step of the UKF"""
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
        pred_mean, pred_cov, _ = _predict(filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}
        return carry, outputs


    # Run the Unscented Kalman Filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def unscented_kalman_smoother(params: ParamsNLGSSM,
                              emissions: Float[Array, "ntime emission_dim"],
                              hyperparams: UKFHyperParams,
                              inputs: Optional[Float[Array, "ntime input_dim"]]=None) \
                              -> PosteriorGSSMSmoothed:
    """Run a unscented Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: array of observations.
        hyperperams: hyper-parameters.
        inputs: optional inputs.

    Returns:
        nlgssm_posterior: posterior object.

    """
    num_timesteps = len(emissions)
    state_dim = params.dynamics_covariance.shape[0]

    # Run the unscented Kalman filter
    ukf_posterior = unscented_kalman_filter(params, emissions, hyperparams, inputs)
    ll = ukf_posterior.marginal_loglik
    filtered_means = ukf_posterior.filtered_means
    filtered_covs = ukf_posterior.filtered_covariances

    # Compute lambda and weights from from hyperparameters
    alpha, beta, kappa = hyperparams.alpha, hyperparams.beta, hyperparams.kappa
    lamb = _compute_lambda(alpha, kappa, state_dim)
    w_mean, w_cov = _compute_weights(state_dim, alpha, beta, lamb)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        """One step of the UKS"""
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Prediction step
        m_pred, S_pred, S_cross = _predict(filtered_mean, filtered_cov, f, Q, lamb, w_mean, w_cov, u)
        G = psd_solve(S_pred, S_cross.T).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the unscented Kalman smoother
    _, (smoothed_means, smoothed_covs) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )
