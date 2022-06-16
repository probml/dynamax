import jax.numpy as jnp
import jax.random as jr
from jax import lax
from distrax import MultivariateNormalFullCovariance as MVN
import chex

@chex.dataclass
class LGSSMInfoParams:
    """Lightweight container for LGSSM parameters in information form.
    """
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
    filtered_etas: chex.Array = None
    filtered_precisions: chex.Array = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim+1 else x


def _info_predict(eta, P, F_inv, Q_prec, B, u, b):
    """Predict next mean and precision under a linear Gaussian model

        p(z_{t+1}) = \int N(z_t | mu_t, Sigma_t) N(z_{t+1} | F z_t, Q)
    """
    I = jnp.eye(len(P))
    # TODO: Is there a way to avoid this inverse?
    Q = jnp.linalg.inv(Q_prec)
    M = F_inv.T @ P @ F_inv
    C = jnp.linalg.solve(I + M @ Q, F_inv.T)
    P_pred = C @ P @ F_inv
    eta_pred = C @ eta + P @ (B @ u + b)
    return eta_pred, P_pred


def _info_condition_on(eta, P, H, R_prec, D, u, d, obs):
    """Condition a Gaussian potential on a new linear Gaussian observation.

        p(z_t|y_t) \prop  N(z_t | mu_{t|t-1}, Sigma_{t|t-1}) N(y_t | H z_t, R)

        The prior precision and precision-weighted mean are given by:
            Lambda_{t|t-1} = Sigma_{t|t-1}^{-1}
            eta_{t|t-1} = Lambda{t|t-1} mu_{t|t-1},
        respectively. 

        The upated parameters are then:
            Lambda_t = Lambda_{t|t-1} + H^T R_prec H
            eta_t = eta_{t|t-1} + H^T R_prec y_t

    Args:
        eta: prior precision weighted mean.
        P: prior precision matrix.
        R_prec: precision matrix for observations.
        obs: observation.
    """
    C = H.T @ R_prec
    P_cond = P + C @ H
    eta_cond = eta + C @ (obs - D @ u - d)
    return eta_cond, P_cond


def lgssm_info_filter(params, inputs, emissions, num_timesteps=None):
    """Run a Kalman filter to produce the marginal likelihood and filtered state
    estimates.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        inputs: array of length T containing inputs.
        emissions: array (T,D) of data.

    Returns:
        filtered_posterior: LGSSMPosterior instance containing,
            filtered_etas
            filtered_precisions
    """
    num_timesteps = len(emissions) if num_timesteps is None else num_timesteps

    def _step(carry, t):
        pred_eta, pred_prec = carry

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

        # Condition on this emission
        filtered_eta, filtered_prec = _info_condition_on(
            pred_eta, pred_prec, H, R_prec, D, u, d, y)

        # Predict the next state
        F_inv = jnp.linalg.inv(F)
        pred_mean, pred_cov = _info_predict(
            filtered_eta, filtered_prec, F_inv, Q_prec, B, u, b)

        return (pred_mean, pred_cov), (filtered_eta, filtered_prec)

    # Run the Kalman filter
    initial_eta = params.initial_precision @ params.initial_mean 
    carry = (initial_eta, params.initial_precision)
    _, (filtered_etas, filtered_precisions) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return LGSSMInfoPosterior(filtered_etas=filtered_etas,
                              filtered_precisions=filtered_precisions)


