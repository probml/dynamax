import jax.numpy as jnp
import jax.random as jr
from jax import lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import chex
from dynamax.containers import GSSMPosterior
from functools import wraps
import inspect


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
    dynamics_covariance: chex.Array
    emission_matrix: chex.Array
    emission_covariance: chex.Array

    # Optional parameters (code below assumes zero otherwise)
    dynamics_input_weights: chex.Array = None
    dynamics_bias: chex.Array  = None
    emission_input_weights: chex.Array = None
    emission_bias: chex.Array = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)

def _predict(m, S, F, B, b, Q, u):
    """Predict next mean and covariance under a linear Gaussian model

        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | Fx_t + Bu + b, Q)
                    = N(x_{t+1} | Fm + Bu, F S F^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        F (D_hid,D_hid): dynamics matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        b (D_hid,): dynamics bias.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    mu_pred = F @ m + B @ u + b
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, H, D, d, R, u, y):
    """Condition a Gaussian potential on a new linear Gaussian observation
       p(x_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t | y_{1:t-1}, u_{1:t-1}) p(y_t | x_t, u_t)
         = N(x_t | m, P) N(y_t | H_t x_t + D_t u_t + d_t, R_t)
         = N(x_t | mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    S = R + H @ P @ H.T
    K = jnp.linalg.solve(S, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - D @ u - d - H @ m)
    return mu_cond, Sigma_cond


def preprocess_args(f):
    """Preprocess the parameters and inputs in case some
    are set to None.

    Args:
        params (_type_): _description_
        num_timesteps (_type_): _description_
        inputs (_type_): _description_
    """
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Extract the arguments by name
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments['params']
        emissions = bound_args.arguments['emissions']
        inputs = bound_args.arguments['inputs']

        # Make sure all the required parameters are there
        assert params.initial_mean is not None
        assert params.initial_covariance is not None
        assert params.dynamics_matrix is not None
        assert params.dynamics_covariance is not None
        assert params.emission_matrix is not None
        assert params.emission_covariance is not None

        # Get shapes
        emission_dim, state_dim = params.emission_matrix.shape[-2:]
        num_timesteps = len(emissions)

        # Default the inputs to zero
        inputs = _zeros_if_none(inputs, (num_timesteps, 0))
        input_dim = inputs.shape[-1]

        # Default other parameters to zero
        dynamics_input_weights = _zeros_if_none(params.dynamics_input_weights, (state_dim, input_dim))
        dynamics_bias = _zeros_if_none(params.dynamics_bias, (state_dim,))
        emission_input_weights = _zeros_if_none(params.emission_input_weights, (emission_dim, input_dim))
        emission_bias = _zeros_if_none(params.emission_bias, (emission_dim,))

        full_params = LGSSMParams(
            initial_mean=params.initial_mean,
            initial_covariance=params.initial_covariance,
            dynamics_matrix=params.dynamics_matrix,
            dynamics_input_weights=dynamics_input_weights,
            dynamics_bias=dynamics_bias,
            dynamics_covariance=params.dynamics_covariance,
            emission_matrix=params.emission_matrix,
            emission_input_weights=emission_input_weights,
            emission_bias=emission_bias,
            emission_covariance=params.emission_covariance
        )
        return f(full_params, emissions, inputs=inputs)
    return wrapper


def lgssm_sample(rng, params, num_timesteps, inputs=None):
    """Sample states and emissions from an LGSSM.

    Args:
        params (_type_): _description_
        num_timesteps (_type_): _description_
        inputs (_type_, optional): _description_. Defaults to None.
    """
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    init_key, rng = jr.split(rng)
    initial_state = MVN(params.initial_mean, params.initial_covariance).sample(seed=init_key)

    def _step(carry, t):
        rng, state = carry
        emission_rng, state_rng, rng = jr.split(rng, 3)
        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        H = _get_params(params.emission_matrix, 2, t)
        D = _get_params(params.emission_input_weights, 2, t)
        d = _get_params(params.emission_bias, 1, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]

        emission = MVN(H @ state + D @ u + d, R).sample(seed=emission_rng)
        next_state = MVN(F @ state + B @ u + b, Q).sample(seed=state_rng)
        return (rng, next_state), (state, emission)

    _, (states, emissions) = lax.scan(_step, (rng, initial_state), jnp.arange(num_timesteps))
    return states, emissions


@preprocess_args
def lgssm_filter(params, emissions, inputs=None):
    """Run a Kalman filter to produce the marginal likelihood and filtered state
    estimates.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        filtered_posterior: GSSMPosterior instance containing,
            marginal_log_lik
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        H = _get_params(params.emission_matrix, 2, t)
        D = _get_params(params.emission_input_weights, 2, t)
        d = _get_params(params.emission_bias, 1, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        ll += MVN(H @ pred_mean + D @ u + d, H @ pred_cov @ H.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return GSSMPosterior(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)




@preprocess_args
def lgssm_smoother(params, emissions, inputs=None):
    """Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        lgssm_posterior: GSSMPosterior instance containing properites of
            filtered and smoothed posterior distributions.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = jnp.linalg.solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]
    return GSSMPosterior(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )


def lgssm_posterior_sample(rng, params, emissions, inputs=None):
    """Run forward-filtering, backward-sampling to draw samples of
        x_{1:T} | y_{1:T}, u_{1:T}.

    Args:
        rng: jax.random.PRNGKey.
        params: an LGSSMParams instance (or object with the same fields)
        emissions (T,D_hid): array of observations.
        inputs (T,D_in): array of inputs.

    Returns:
        ll: marginal log likelihood of the observations.
        states (T,D_hid): samples from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior.to_tuple()

    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        rng, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics_matrix, 2, t)
        B = _get_params(params.dynamics_input_weights, 2, t)
        b = _get_params(params.dynamics_bias, 1, t)
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, F, B, b, Q, u, next_state)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=rng)
        return state, state

    # Initialize the last state
    rng, this_rng = jr.split(rng, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_rng)

    args = (
        jr.split(rng, num_timesteps - 1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
        jnp.arange(num_timesteps - 2, -1, -1),
    )
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return ll, states