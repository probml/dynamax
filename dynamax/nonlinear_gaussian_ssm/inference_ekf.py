import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional

from dynamax.utils.utils import psd_solve, symmetrize
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.types import PRNGKey

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x


def _predict(m, P, f, F, Q, u):
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

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
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
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
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, symmetrize(Sigma_cond)


def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

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

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def iterated_extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMFiltered:
    r"""Run an iterated extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    filtered_posterior = extended_kalman_filter(params, emissions, num_iter, inputs)
    return filtered_posterior


def extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
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
        G = psd_solve(S_pred, F_x @ filtered_cov).T

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
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )




def extended_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples.

    Args:
        key: random number key.
        params: model parameters.
        emissions: observation sequence.
        inputs: optional array of inputs.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    filtered_posterior = extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, f, F, Q, u, next_state, 1)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    args = (
        jr.split(key, num_timesteps - 1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
        jnp.arange(num_timesteps - 2, -1, -1),
    )
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return states



def iterated_extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an iterated extended Kalman smoother (IEKS).

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """

    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = extended_kalman_smoother(params, emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior
