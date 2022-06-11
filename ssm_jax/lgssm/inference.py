import jax.numpy as jnp
import jax.random as jr
from jax import lax
from distrax import MultivariateNormalFullCovariance as MVN
import chex

@chex.dataclass
class LGSSMParams:
    """Lightweight container for LGSSM parameters.
    The functions below can be called with an instance of this class.
    However, they can also accept a ssm.lgssm.models.LinearGaussianSSM instance,
    if you prefer a more object-oriented approach.
    """
    initial_mean: chex.Array
    initial_covariance: chex.Array
    dynamics_matrix: chex.Array
    dynamics_input_weights: chex.Array
    dynamics_bias: chex.Array
    dynamics_covariance: chex.Array
    emission_matrix: chex.Array
    emission_input_weights: chex.Array
    emission_bias: chex.Array
    emission_covariance: chex.Array

@chex.dataclass
class LGSSMPosterior:
    """Simple wrapper for properties of an LGSSM posterior distribution.

    Attributes:
            marginal_loglik: marginal log likelihood of the data
            filtered_means: (T,K) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_covariances: (T,K,K) array,
                Cov[x_t | y_{1:t}, u_{1:t}].
            smoothed_means: (T,K) array,
                E[x_t | y_{1:T}, u_{1:T}].
            smoothed_covs: (T,K,K) array of smoothed marginal covariances,
                Cov[x_t | y_{1:T}, u_{1:T}].
            smoothed_cross: (T-1, K, K) array of smoothed cross products,
                E[x_t x_{t+1}^T | y_{1:T}, u_{1:T}].
    """
    marginal_loglik: chex.Scalar = None
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None
    smoothed_cross_covariances: chex.Array = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim+1 else x

def _predict(m, S, A, B, b, Q, u):
    """Predict next mean and covariance under a linear Gaussian model

        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | Ax_t + Bu + b, Q)
                    = N(x_{t+1} | Am + Bu, A S A^T + Q)
    """
    mu_pred = A @ m + B @ u + b
    Sigma_pred = A @ S @ A.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, S, C, D, d, R, u, y):
    """Condition a Gaussian potential on a new linear Gaussian observation

    **Note! This can be done more efficiently when R is diagonal.**
    """
    # Compute the Kalman gain
    K = jnp.linalg.solve(R + C @ S @ C.T, C @ S).T

    # Follow equations 8.80 and 8.86 in PML2
    # This should be more numerically stable than
    # Sigma_cond = S - K @ C @ S
    dim = m.shape[-1]
    ImKC = jnp.eye(dim) - K @ C
    Sigma_cond = ImKC @ S @ ImKC.T + K @ R @ K.T
    mu_cond = m + K @ (y - D @ u - d - C @ m)
    return mu_cond, Sigma_cond


def lgssm_filter(params, emissions, inputs=None, num_timesteps=None):
    """Run a Kalman filter to produce the marginal likelihood and filtered state
    estimates.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        inputs: array of length T containing inputs.
        emissions: array (T,D) of data.

    Returns:
        filtered_posterior: LGSSMPosterior instance containing,
            marginal_log_lik
            filtered_means
            filtered_covariances
    """
    num_timesteps = len(emissions) if num_timesteps is None else num_timesteps
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        A = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        C = _get_params(params.emission_matrix, 2, t)
        D = _get_params(params.emission_input_weights, 2, t)
        d = _get_params(params.emission_bias, 1, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        ll += MVN(C @ pred_mean + D @ u + d,
                  C @ pred_cov @ C.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, C, D, d, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(
            filtered_mean, filtered_cov, A, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0., params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return LGSSMPosterior(marginal_loglik=ll,
                          filtered_means=filtered_means,
                          filtered_covariances=filtered_covs)


def lgssm_posterior_sample(rng, params, emissions, inputs=None, num_timesteps=None):
    """Run forward-filtering, backward-sampling to draw samples of
        x_{1:T} | y_{1:T}, u_{1:T}.

    Args:
        rng: jax.random.PRNGKey.
        params: an LGSSMParams instance (or object with the same fields).
        inputs: array of length T containing inputs.
        emissions: array (T,D) of data.

    Returns:
        ll: marginal log likelihood of the data
        states: array (T,K) of samples from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions) if num_timesteps is None else num_timesteps
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, inputs, emissions, num_timesteps)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        rng, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        A = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(
            filtered_mean, filtered_cov, A, B, b, Q, u, next_state)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=rng)
        return state, state

    # Initialize the last state
    rng, this_rng = jr.split(rng, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_rng)

    args = (jr.split(rng, num_timesteps-1),
            filtered_means[:-1][::-1],
            filtered_covs[:-1][::-1],
            jnp.arange(num_timesteps-2, -1, -1))
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return ll, states


def lgssm_smoother(params, emissions, inputs=None, num_timesteps=None):
    """Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        inputs: array of (T,Din) containing inputs.
        emissions: array (T,Dout) of data.

    Returns:
        lgssm_posterior: LGSSMPosterior instance containing properites of
            filtered and smoothed posterior distributions.
    """
    num_timesteps = len(emissions) if num_timesteps is None else num_timesteps
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, inputs, emissions, num_timesteps)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        A = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = jnp.linalg.solve(Q + A @ filtered_cov @ A.T, A @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + \
            G @ (smoothed_mean_next - A @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + \
            G @ (smoothed_cov_next - A @ filtered_cov @ A.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + \
            jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), \
               (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps-2, -1, -1),
            filtered_means[:-1][::-1],
            filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs, smoothed_cross) = \
        lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None,...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None,...]))
    smoothed_cross = smoothed_cross[::-1]
    return LGSSMPosterior(marginal_loglik=ll,
                          filtered_means=filtered_means,
                          filtered_covariances=filtered_covs,
                          smoothed_means=smoothed_means,
                          smoothed_covariances=smoothed_covs,
                          smoothed_cross_covariances=smoothed_cross)
