import jax.numpy as jnp
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import chex


@chex.dataclass
class LGSSMInfoParams:
    """Lightweight container for LGSSM parameters in information form."""

    initial_mean: chex.Array
    initial_precision: chex.Array
    dynamics_matrix: chex.Array
    dynamics_precision: chex.Array
    dynamics_input_weights: chex.Array
    dynamics_bias: chex.Array
    emission_matrix: chex.Array
    emission_input_weights: chex.Array
    emission_bias: chex.Array
    emission_precision: chex.Array


@chex.dataclass
class LGSSMInfoPosterior:
    """Simple wrapper for properties of an LGSSM posterior distribution in
    information form.

    Attributes:
            filtered_means: (T,K) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_precisions: (T,K,K) array,
                inv(Cov[x_t | y_{1:t}, u_{1:t}]).
    """

    marginal_loglik: chex.Scalar = None
    filtered_etas: chex.Array = None
    filtered_precisions: chex.Array = None
    smoothed_etas: chex.Array = None
    smoothed_precisions: chex.Array = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x


def _mvn_info_log_prob(eta, Lambda, x):
    """Calculate the log probability of an observation from a MVN
    parameterised in information form.

    Args:
        eta (D,): precision weighted mean.
        Lambda (D,D): precision.
        x (D,): observation.

    Returns:
        log_prob: log probability of x.
    """
    D = len(Lambda)
    lp = x.T @ eta - 0.5 * x.T @ Lambda @ x
    lp += -0.5 * eta.T @ jnp.linalg.solve(Lambda, eta)
    sign, logdet = jnp.linalg.slogdet(Lambda)
    lp += -0.5 * (D * jnp.log(2 * jnp.pi) - sign * logdet)
    return lp


def _info_predict(eta, Lambda, F, Q_prec, B, u, b):
    """Predict next mean and precision under a linear Gaussian model.

    Marginalising over the uncertainty in z_t the predicted latent state at
    the next time step is given by:
        p(z_{t+1}| z_t, u_t)
            = \int p(z_{t+1}, z_t | u_t) dz_t
            = \int N(z_t | mu_t, Sigma_t) N(z_{t+1} | F z_t + B u_t + b, Q) dz_t
            = N(z_t | m_{t+1|t}, Sigma_{t+1|t})
    with
        m_{t+1|t} = F m_t + B u_t + b
        Sigma_{t+1|t} = F Sigma_t F^T + Q

    The corresponding information form parameters are:
        eta_{t+1|t} = K eta_t + Lambda_{t+1|t} (B u_t + b)
        Lambda_{t+1|t} = L Q_prec L^T + K Lambda_t K^T
    where
        K = Q_prec F ( Lambda_t + F^T Q_prec F)^{-1}
        L = I - K F^T

    Args:
        eta (D_hid,): prior precision weighted mean.
        Lambda (D_hid,D_hid): prior precision matrix.
        F (D_hid,D_hid): dynamics matrix.
        Q_prec (D_hid,D_hid): dynamics precision matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        b (D_hid,): dynamics bias.

    Returns:
        eta_pred (D_hid,): predicted precision weighted mean.
        Lambda_pred (D_hid,D_hid): predicted precision.
    """
    K = jnp.linalg.solve(Lambda + F.T @ Q_prec @ F, F.T @ Q_prec).T
    I = jnp.eye(F.shape[0])
    ## This version should be more stable than:
    # Lambda_pred = (I - K @ F.T) @ Q_prec
    ImKF = I - K @ F.T
    Lambda_pred = ImKF @ Q_prec @ ImKF.T + K @ Lambda @ K.T
    eta_pred = K @ eta + Lambda_pred @ (B @ u + b)
    return eta_pred, Lambda_pred


def _info_condition_on(eta, Lambda, H, R_prec, D, u, d, obs):
    """Condition a Gaussian potential on a new linear Gaussian observation.

        p(z_t|y_t, u_t) \prop  N(z_t | mu_{t|t-1}, Sigma_{t|t-1}) *
                          N(y_t | H z_t + D u_t + d, R)

    The prior precision and precision-weighted mean are given by:
        Lambda_{t|t-1} = Sigma_{t|t-1}^{-1}
        eta_{t|t-1} = Lambda{t|t-1} mu_{t|t-1},
    respectively.

    The upated parameters are then:
        Lambda_t = Lambda_{t|t-1} + H^T R_prec H
        eta_t = eta_{t|t-1} + H^T R_prec (y_t - Du - d)

    Args:
        eta (D_hid,): prior precision weighted mean.
        Lambda (D_hid,D_hid): prior precision matrix.
        H (D_obs,D_hid): emission matrix.
        R_prec (D_obs,D_obs): precision matrix for observations.
        D (D_obs,D_in): emission input weights.
        u (D_in,): inputs.
        d (D_obs,): emission bias.
        obs (D_obs,): observation.

    Returns:
        eta_cond (D_hid,): posterior precision weighted mean.
        Lambda_cond (D_hid,D_hid): posterior precision.
    """
    HR = H.T @ R_prec
    Lambda_cond = Lambda + HR @ H
    eta_cond = eta + HR @ (obs - D @ u - d)
    return eta_cond, Lambda_cond


def lgssm_info_filter(params, emissions, inputs):
    """Run a Kalman filter to produce the filtered state estimates.

    Args:
        params: an LGSSMInfoParams instance.
        emissions (T,D_obs): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: LGSSMInfoPosterior instance containing,
            filtered_etas
            filtered_precisions
    """
    num_timesteps = len(emissions)

    def _filter_step(carry, t):
        ll, pred_eta, pred_prec = carry

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        Q_prec = _get_params(params.dynamics_precision, 2, t)
        H = _get_params(params.emission_matrix, 2, t)
        R_prec = _get_params(params.emission_precision, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        D = _get_params(params.emission_input_weights, 2, t)
        d = _get_params(params.emission_bias, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        y_pred_eta, y_pred_prec = _info_predict(pred_eta, pred_prec, H, R_prec, D, u, d)
        ll += _mvn_info_log_prob(y_pred_eta, y_pred_prec, y)

        # Condition on this emission
        filtered_eta, filtered_prec = _info_condition_on(pred_eta, pred_prec, H, R_prec, D, u, d, y)

        # Predict the next state
        pred_eta, pred_prec = _info_predict(filtered_eta, filtered_prec, F, Q_prec, B, u, b)

        return (ll, pred_eta, pred_prec), (filtered_eta, filtered_prec)

    # Run the Kalman filter
    initial_eta = params.initial_precision @ params.initial_mean
    carry = (0.0, initial_eta, params.initial_precision)
    (ll, _, _), (filtered_etas, filtered_precisions) = lax.scan(_filter_step, carry, jnp.arange(num_timesteps))
    return LGSSMInfoPosterior(marginal_loglik=ll, filtered_etas=filtered_etas, filtered_precisions=filtered_precisions)


def lgssm_info_smoother(params, emissions, inputs=None):
    """Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. This
    is the information form of the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an LGSSMInfoParams instance.
        inputs: array of (T,Din) containing inputs.
        emissions: array (T,Dout) of data.

    Returns:
        lgssm_info_posterior: LGSSMInfoPosterior instance containing properites
            of filtered and smoothed posterior distributions.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_info_filter(params, emissions, inputs)
    ll, filtered_etas, filtered_precisions, *_ = filtered_posterior.to_tuple()

    # Run the smoother backward in time
    def _smooth_step(carry, args):
        # Unpack the inputs
        smoothed_eta_next, smoothed_prec_next = carry
        t, filtered_eta, filtered_prec = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q_prec = _get_params(params.dynamics_precision, 2, t)
        u = inputs[t]

        # Predict the next state
        # TODO: Pass predicted params from lgssm_info_filter?
        pred_eta, pred_prec = _info_predict(filtered_eta, filtered_prec, F, Q_prec, B, u, b)

        # This is the information form version of the 'reverse' Kalman gain
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = jnp.linalg.solve(Q_prec + smoothed_prec_next - pred_prec, Q_prec @ F)

        # Compute the smoothed parameter estimates
        smoothed_prec = filtered_prec + F.T @ Q_prec @ (F - G)
        smoothed_eta = filtered_eta + G.T @ (smoothed_eta_next - pred_eta) + (G.T - F.T) @ Q_prec @ (B @ u + b)

        return (smoothed_eta, smoothed_prec), (smoothed_eta, smoothed_prec)

    # Run the Kalman smoother
    init_carry = (filtered_etas[-1], filtered_precisions[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_etas[:-1][::-1], filtered_precisions[:-1][::-1])
    _, (smoothed_etas, smoothed_precisions) = lax.scan(_smooth_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_etas = jnp.row_stack((smoothed_etas[::-1], filtered_etas[-1][None, ...]))
    smoothed_precisions = jnp.row_stack((smoothed_precisions[::-1], filtered_precisions[-1][None, ...]))
    return LGSSMInfoPosterior(
        marginal_loglik=ll,
        filtered_etas=filtered_etas,
        filtered_precisions=filtered_precisions,
        smoothed_etas=smoothed_etas,
        smoothed_precisions=smoothed_precisions,
    )
