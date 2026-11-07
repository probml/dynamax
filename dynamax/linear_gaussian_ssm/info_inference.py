"""
Inference algorithms for linear Gaussian state space models in information form.
"""
import jax.numpy as jnp

from dynamax.linear_gaussian_ssm.block_tridiag import block_tridiag_mvn_filter
from dynamax.utils.utils import psd_solve
from jax import lax, vmap, value_and_grad
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Tuple


class ParamsLGSSMInfo(NamedTuple):
    """Lightweight container for passing LGSSM parameters in information form to inference algorithms."""
    initial_mean: Float[Array, " state_dim"]
    dynamics_weights: Float[Array, "state_dim state_dim"]
    emission_weights:  Float[Array, "emission_dim state_dim"]

    initial_precision: Float[Array, "state_dim state_dim"]
    dynamics_precision:  Float[Array, "state_dim state_dim"]
    emission_precision: Float[Array, "emission_dim emission_dim"]

    # Optional parameters (None means zeros)
    dynamics_input_weights: Optional[Float[Array, "input_dim state_dim"]] = None
    dynamics_bias: Optional[Float[Array, " state_dim"]] = None
    emission_input_weights: Optional[Float[Array, "input_dim emission_dim"]] = None
    emission_bias: Optional[Float[Array, " emission_dim"]] = None


class PosteriorGSSMInfoFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior in information form.

    Attributes:
        marginal_loglik
        filtered_means: (T,K) array,
            E[x_t \mid y_{1:t}, u_{1:t}].
        filtered_precisions: (T,K,K) array,
            inv(Cov[x_t \mid y_{1:t}, u_{1:t}]).
    """
    marginal_loglik: Float[Array, ""] # Scalar
    filtered_etas: Float[Array, "ntime state_dim"]
    filtered_precisions: Float[Array, "ntime state_dim state_dim"]


class PosteriorGSSMInfoSmoothed(NamedTuple):
    """"Marginals of the Gaussian filtering and smoothed posterior in information form.
    """
    marginal_loglik: Float[Array, ""] # Scalar
    filtered_etas: Float[Array, "ntime state_dim"]
    filtered_precisions: Float[Array, "ntime state_dim state_dim"]
    smoothed_etas: Float[Array, "ntime state_dim"]
    smoothed_precisions: Float[Array, "ntime state_dim state_dim"]

def info_to_moment_form(etas, Lambdas):
    """Convert information form parameters to moment form.

    Args:
        etas (N,D): precision weighted means.
        Lambdas (N,D,D): precision matrices.

    Returns:
        means (N,D)
        covs (N,D,D)
    """
    means = vmap(lambda A, b:psd_solve(A, b))(Lambdas, etas)
    covs = jnp.linalg.inv(Lambdas)
    return means, covs

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
    lp += -0.5 * eta.T @ psd_solve(Lambda, eta)
    sign, logdet = jnp.linalg.slogdet(Lambda)
    lp += -0.5 * (D * jnp.log(2 * jnp.pi) - sign * logdet)
    return lp


def _info_predict(eta, Lambda, F, Q_prec, B, u, b):
    r"""Predict next mean and precision under a linear Gaussian model.

    Marginalising over the uncertainty in z_t the predicted latent state at
    the next time step is given by:
        p(z_{t+1}\mid z_t, u_{t+1})
            = \int p(z_{t+1}, z_t \mid u_{t+1}) dz_t
            = \int N(z_t \mid mu_t, Sigma_t) N(z_{t+1} \mid F z_t + B u_{t+1} + b, Q) dz_t
            = N(z_t \mid m_{t+1\midt}, Sigma_{t+1\mid t})
    with
        m_{t+1 \mid t} = F m_t + B u_{t+1} + b
        Sigma_{t+1 \mid t} = F Sigma_t F^T + Q

    The corresponding information form parameters are:
        eta_{t+1 \mid t} = K eta_t + Lambda_{t+1 \mid t} (B u_{t+1} + b)
        Lambda_{t+1 \mid t} = L Q_prec L^T + K Lambda_t K^T
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
    K = psd_solve(Lambda + F.T @ Q_prec @ F, F.T @ Q_prec).T
    I = jnp.eye(F.shape[0])
    ## This version should be more stable than:
    # Lambda_pred = (I - K @ F.T) @ Q_prec
    ImKF = I - K @ F.T
    Lambda_pred = ImKF @ Q_prec @ ImKF.T + K @ Lambda @ K.T
    eta_pred = K @ eta + Lambda_pred @ (B @ u + b)
    return eta_pred, Lambda_pred


def _info_condition_on(eta, Lambda, H, R_prec, D, u, d, obs):
    r"""Condition a Gaussian potential on a new linear Gaussian observation.

        p(z_t \mid y_t, u_t) \prop  N(z_t  \mid  mu_{t \mid t-1}, Sigma_{t \mid t-1}) *
                          N(y_t  \mid  H z_t + D u_t + d, R)

    The prior precision and precision-weighted mean are given by:
        Lambda_{t \mid t-1} = Sigma_{t \mid t-1}^{-1}
        eta_{t \mid t-1} = Lambda{t \mid t-1} mu_{t \mid t-1},
    respectively.

    The upated parameters are then:
        Lambda_t = Lambda_{t \mid t-1} + H^T R_prec H
        eta_t = eta_{t \mid t-1} + H^T R_prec (y_t - Du - d)

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


def lgssm_info_filter(
    params: ParamsLGSSMInfo,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMInfoFiltered:
    r"""Run a Kalman filter to produce the filtered state estimates.

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
    for param, ndim in ((params.dynamics_weights, 2), (params.dynamics_input_weights, 2),
                        (params.dynamics_bias, 1), (params.dynamics_precision, 2)):
        if param is not None and not callable(param) and param.ndim == ndim + 1:
            assert param.shape[0] == num_timesteps
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs
    def _filter_step(carry, t):
        """Run a single step of the Kalman filter."""
        ll, pred_eta, pred_prec = carry

        # Shorthand: get parameters and inputs for time index t
        H = _get_params(params.emission_weights, 2, t)
        R_prec = _get_params(params.emission_precision, 2, t)
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
        t_next = jnp.minimum(t + 1, num_timesteps - 1)
        F = _get_params(params.dynamics_weights, 2, t_next)
        Q_prec = _get_params(params.dynamics_precision, 2, t_next)
        B = _get_params(params.dynamics_input_weights, 2, t_next)
        b = _get_params(params.dynamics_bias, 1, t_next)
        pred_eta, pred_prec = _info_predict(filtered_eta, filtered_prec, F, Q_prec, B, inputs[t_next], b)

        return (ll, pred_eta, pred_prec), (filtered_eta, filtered_prec)

    # Run the Kalman filter
    initial_eta = params.initial_precision @ params.initial_mean
    carry = (0.0, initial_eta, params.initial_precision)
    (ll, _, _), (filtered_etas, filtered_precisions) = lax.scan(_filter_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMInfoFiltered(marginal_loglik=ll, filtered_etas=filtered_etas, filtered_precisions=filtered_precisions)


def lgssm_info_smoother(
    params: ParamsLGSSMInfo,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMInfoSmoothed:
    r"""Run forward-filtering, backward-smoother to compute expectations
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
    ll, filtered_etas, filtered_precisions, *_ = filtered_posterior

    # Run the smoother backward in time
    def _smooth_step(carry, args):
        """Run a single step of the Kalman smoother."""
        # Unpack the inputs
        smoothed_eta_next, smoothed_prec_next = carry
        t, filtered_eta, filtered_prec = args

        # Shorthand: get parameters and inputs for time index t+1
        F = _get_params(params.dynamics_weights, 2, t + 1)
        B = _get_params(params.dynamics_input_weights, 2, t + 1)
        b = _get_params(params.dynamics_bias, 1, t + 1)
        Q_prec = _get_params(params.dynamics_precision, 2, t + 1)
        u = inputs[t + 1]

        # Predict the next state
        # TODO: Pass predicted params from lgssm_info_filter?
        pred_eta, pred_prec = _info_predict(filtered_eta, filtered_prec, F, Q_prec, B, u, b)

        # This is the information form version of the 'reverse' Kalman gain
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = psd_solve(Q_prec + smoothed_prec_next - pred_prec, Q_prec @ F)

        # Compute the smoothed parameter estimates
        smoothed_prec = filtered_prec + F.T @ Q_prec @ (F - G)
        smoothed_eta = filtered_eta + G.T @ (smoothed_eta_next - pred_eta) + (G.T - F.T) @ Q_prec @ (B @ u + b)

        return (smoothed_eta, smoothed_prec), (smoothed_eta, smoothed_prec)

    # Run the Kalman smoother
    _, (smoothed_etas, smoothed_precisions) = lax.scan(
        _smooth_step,
        (filtered_etas[-1], filtered_precisions[-1]),
        (jnp.arange(num_timesteps - 1), filtered_etas[:-1], filtered_precisions[:-1]),
        reverse=True
    )

    # Concatenate the arrays and return
    smoothed_etas = jnp.vstack((smoothed_etas, filtered_etas[-1][None, ...]))
    smoothed_precisions = jnp.vstack((smoothed_precisions, filtered_precisions[-1][None, ...]))

    return PosteriorGSSMInfoSmoothed(
        marginal_loglik=ll,
        filtered_etas=filtered_etas,
        filtered_precisions=filtered_precisions,
        smoothed_etas=smoothed_etas,
        smoothed_precisions=smoothed_precisions,
    )


def block_tridiag_mvn_log_normalizer(precision_diag_blocks: Float[Array, "num_timesteps state_dim state_dim"], 
                                     precision_lower_diag_blocks: Float[Array, "num_timesteps-1 state_dim state_dim"], 
                                     linear_potential: Float[Array, "num_timesteps state_dim"]) \
                                     -> Tuple[Float, Tuple[Float[Array, "num_timesteps state_dim state_dim"], 
                                                           Float[Array, "num_timesteps state_dim"]]]:
    r"""
    Compute the log normalizing constant for a multivariate normal distribution
    with natural parameters :math:`J` and :math:`h` with density,
    ..math:
        \log p(x) = -1/2 x^\top J x + h^\top x - \log Z

    where the log normalizer is
    ..math:
        \log Z = N/2 \log 2 \pi - 1/2 \log |J| + 1/2 h^T J^{-1} h

    and :math:`N` is the dimensionality.

    Typically, computing the log normalizer is cubic in N, but if :math:`J` is
    block tridiagonal, it can be computed in O(N) time. Specifically, suppose
    J is TDxTD with blocks of size D on the diagonal and first off diagonal.
    Since J is symmetric, we can represent the matrix by only its diagonal and
    first lower diagonal blocks. This is exactly the type of precision matrix
    we encounter with linear Gaussian dynamical systems. This code computes its
    log normalizer using the so-called "information form Kalman filter."

    Args:

    precision_diag_blocks:          Shape (T, D, D) array of the diagonal blocks
                                    of a shape (TD, TD) precision matrix.
    precision_lower_diag_blocks:    Shape (T-1, D, D) array of the lower diagonal
                                    blocks of a shape (TD, TD) precision matrix.
    linear_potential:               Shape (T, D) array of linear potentials of a
                                    TD dimensional multivariate normal distribution
                                    in information form.

    Returns:

    log_normalizer:                 The scalar log normalizing constant.
    (filtered_Js, filtered_hs):     The precision and linear potentials of the
                                    Gaussian filtering distributions in information
                                    form, with shape (T, D, D) and (T, D) respectively.
    """
    log_normalizer, _, filtered_Js, filtered_hs = block_tridiag_mvn_filter(
        precision_diag_blocks, precision_lower_diag_blocks, linear_potential)
    return log_normalizer, (filtered_Js, filtered_hs)


def block_tridiag_mvn_expectations(precision_diag_blocks: Float[Array, "num_timesteps state_dim state_dim"], 
                                   precision_lower_diag_blocks: Float[Array, "num_timesteps-1 state_dim state_dim"], 
                                   linear_potential: Float[Array, "num_timesteps state_dim"]) \
                                   -> Tuple[Float, 
                                            Float[Array, "num_timesteps state_dim"],
                                            Float[Array, "num_timesteps state_dim state_dim"],
                                            Float[Array, "num_timesteps-1 state_dim state_dim"]]:
    """
    Compute the posterior expectations of a multivariate normal distribution
    with block tridiagonal precision matrix in O(T) time using the
    information form Kalman filter.
    
    Note: this implementation uses automatic differentiation of the log normalizer
    to compute the expected sufficient statistics. 

    Args:

    precision_diag_blocks:          Shape (T, D, D) array of the diagonal blocks
                                    of a shape (TD, TD) precision matrix.
    precision_lower_diag_blocks:    Shape (T-1, D, D) array of the lower diagonal
                                    blocks of a shape (TD, TD) precision matrix.
    linear_potential:               Shape (T, D) array of linear potentials of a
                                    TD dimensional multivariate normal distribution

    Returns:
    
    log_normalizer:                 The scalar log normalizing constant.
    Ex:                             The expected value of x, with shape (T, D).
    ExxT:                           The expected value of x x^T, with shape (T, D, D).
    ExxnT:                          The expected value of x_{t+1} x_t^T, with shape (T-1, D, D).

    """
    # Run message passing code to get the log normalizer, the filtering potentials,
    # and the expected values of x. Technically, the natural parameters are -1/2 J
    # so we need to do a little correction of the gradients to get the expectations.
    f = value_and_grad(block_tridiag_mvn_log_normalizer, argnums=(0, 1, 2), has_aux=True)
    (log_normalizer, _), grads = f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential)

    # Correct for the -1/2 J -> J implementation
    ExxT = -2 * grads[0]
    ExxnT = -grads[1]
    Ex = grads[2]
    return log_normalizer, Ex, ExxT, ExxnT

