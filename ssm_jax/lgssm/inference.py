import jax.numpy as jnp
import jax.random as jr
from jax import lax
from distrax import MultivariateNormalFullCovariance as MVN

# Helper functions
def _predict(m, S, A, B, Q, u):
    """
        Predict next mean and covariance under a linear Gaussian model

        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | Ax_t + Bu, Q)
                    = N(x_{t+1} | Am + Bu, A S A^T + Q)
    """
    mu_pred = A @ m + B @ u
    Sigma_pred = A @ S @ A.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, S, C, D, R, u, y):
    """
    Condition a Gaussian potential on a new linear Gaussian observation

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
    mu_cond = m + K @ (y - D @ u - C @ m)
    return mu_cond, Sigma_cond


def lgssm_filter(params, inputs, emissions):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state
    estimates.

    Args:

    lds_params: an LDSParams instance
    inputs: array of inputs to the LDS
    observations: array of data

    Returns:

    ll:             marginal log likelihood of the data
    filtered_means: filtered means E[x_t | y_{1:t}, u_{1:t}]
    filtered_covs:  filtered covariances Cov[x_t | y_{1:t}, u_{1:t}]
    """

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(params.dynamics_matrix)
        B = get(params.dynamics_input_weights)
        Q = get(params.dynamics_covariance)
        C = get(params.emission_matrix)
        D = get(params.emission_input_weights)
        R = get(params.emission_covariance)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        ll += MVN(C @ pred_mean + D @ u,
                  C @ pred_cov @ C.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(
            pred_mean, pred_cov, C, D, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(
            filtered_mean, filtered_cov, A, B, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    num_timesteps = len(emissions)
    carry = (0., params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return ll, filtered_means, filtered_covs


def lgssm_posterior_sample(rng, params, inputs, emissions):
    """
    Run forward-filtering, backward-sampling to draw samples of
        x_{1:T} | y_{1:T}, u_{1:T}.

    Args:

    rng:        jax.random.PRNGKey
    lds:        an LDSParams instance
    inputs:     array of inputs to the LDS
    data:       array of data

    Returns:

    ll:         marginal log likelihood of the data
    xs:         samples from the posterior distribution on latent states.
    """
    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(params, inputs, emissions)

    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        rng, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(params.dynamics_matrix)
        B = get(params.dynamics_input_weights)
        Q = get(params.dynamics_covariance)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(
            filtered_mean, filtered_cov, A, B, Q, u, next_state)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=rng)
        return state, state

    # Initialize the last state
    rng, this_rng = jr.split(rng, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_rng)

    num_timesteps = len(emissions)
    args = (jr.split(rng, num_timesteps-1),
            filtered_means[:-1][::-1],
            filtered_covs[:-1][::-1],
            jnp.arange(num_timesteps-2, -1, -1))
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return ll, states


def lgssm_smoother(params, inputs, emissions):
    """Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        lds (_type_): _description_
        inputs (_type_): _description_
        data (_type_): _description_

    Returns:
        ll: marginal log likelihood of the data
        smoothed_means: smoothed mean of the latent states.
        smoothed_covs: smoothed marginal covariance of the latent states.
        smoothed_cross: smoothed cross product E[x_t x_{t+1}^T | y_{1:T}].
    """

    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(params, inputs, emissions)

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(params.dynamics_matrix)
        B = get(params.dynamics_input_weights)
        Q = get(params.dynamics_covariance)
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = jnp.linalg.solve(Q + A @ filtered_cov @ A.T, A @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + \
            G @ (smoothed_mean_next - A @ filtered_mean - B @ u)
        smoothed_cov = filtered_cov + \
            G @ (smoothed_cov_next - A @ filtered_cov @ A.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + \
            jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), \
               (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    num_timesteps = len(emissions)
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
    return ll, smoothed_means, smoothed_covs, smoothed_cross
