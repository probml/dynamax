import jax.numpy as np
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
    K = np.linalg.solve(R + C @ S @ C.T, C @ S).T

    # Sigma_cond = S - K @ C @ S
    # mu_cond = Sigma_cond @ (np.linalg.solve(S, m) +
    #                         C.T @ np.linalg.solve(R, y - D @ u))

    # Follow equations 8.80 and 8.86 in PML2
    # This should be more numerically stable than
    # Sigma_cond = S - K @ C @ S
    dim = m.shape[-1]
    ImKC = np.eye(dim) - K @ C
    Sigma_cond = ImKC @ S @ ImKC.T + K @ R @ K.T
    mu_cond = m + K @ (y - D @ u - C @ m)
    return mu_cond, Sigma_cond


def lds_filter(lds, inputs, data):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state
    estimates.

    Args:

    lds:        an LDS-like object (e.g. ssm_jax.lglds.models.LinearGaussianSSM)
    inputs:     array of inputs to the LDS
    data:       array of data

    Returns:

    ll:             marginal log likelihood of the data
    filtered_means: filtered means E[x_t | y_{1:t}, u_{1:t}]
    filtered_covs:  filtered covariances Cov[x_t | y_{1:t}, u_{1:t}]
    """
    T = len(data)

    def _step(carry, t):
        ll, mu_ttm1, Sigma_ttm1 = carry

        # Get parameters and inputs for time index t
        At = lds.dynamics_matrix(t)
        Bt = lds.dynamics_input_weights(t)
        Qt = lds.dynamics_covariance(t)
        Ct = lds.emissions_matrix(t)
        Dt = lds.emissions_input_weights(t)
        Rt = lds.emissions_covariance(t)
        ut = inputs[t]
        yt = data[t]

        # Update the log likelihood
        ll += MVN(Ct @ mu_ttm1 + Dt @ ut,
                  Ct @ Sigma_ttm1 @ Ct.T + Rt).log_prob(yt)

        # Condition on this frame's observations
        # TODO: This can be more efficient when R is diagonal
        mu_tt, Sigma_tt = _condition_on(
            mu_ttm1, Sigma_ttm1, Ct, Dt, Rt, ut, yt)

        # Predict the next frame's latent state
        mu_tp1t, Sigma_tp1t = _predict(
            mu_tt, Sigma_tt, At, Bt, Qt, ut)

        return (ll, mu_tp1t, Sigma_tp1t), (mu_tt, Sigma_tt)

    # Initialize
    carry = (0., lds.m0, lds.Q0)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(
        _step, carry, np.arange(T)
    )
    return ll, filtered_means, filtered_covs


def lds_posterior_sample(rng, lds, inputs, data):
    """
    Run forward-filtering, backward-sampling to draw samples of
        x_{1:T} | y_{1:T}, u_{1:T}.

    Args:

    rng:        jax.random.PRNGKey
    lds:        an LDS-like object (e.g. ssm_jax.lglds.models.LDS)
    inputs:     array of inputs to the LDS
    data:       array of data

    Returns:

    ll:         marginal log likelihood of the data
    xs:         samples from the posterior distribution on latent states.
    """
    T = len(data)

    # Run the Kalman Filter
    ll, filtered_means, filterd_covs = lds_filter(lds, inputs, data)

    # Sample backward in time
    def _step(carry, args):
        xtp1 = carry
        rng, mu_filt, Sigma_filt, t = args

        # Get parameters and inputs for time index t
        At = lds.dynamics_matrix(t)
        Bt = lds.dynamics_input_weights(t)
        Qt = lds.dynamics_covariance(t)
        ut = inputs[t]

        # Condition on x[t+1]
        mu_post, Sigma_post = _condition_on(
            mu_filt, Sigma_filt, At, Bt, Qt, ut, xtp1)
        xt = MVN(mu_post, Sigma_post).sample(seed=rng)
        return xt, xt

    # Initialize the last state
    rng, this_rng = jr.split(rng, 2)
    xT = MVN(filtered_means[-1], filterd_covs[-1]).sample(seed=this_rng)

    # TODO: Double check the indexing here! We should be ok since all the As,
    # Bs, us, etc are the same, but when those change its easy to get off-by-one
    # indexing bugs. Make sure the right parameters and inputs go to the right
    # time steps.
    args = (jr.split(rng, T-1),
            filtered_means[:-1][::-1],
            filterd_covs[:-1][::-1],
            np.arange(T-1, -1, -1))
    _, xs = lax.scan(_step, xT, args)
    xs = np.row_stack([xs[::-1], xT])
    return ll, xs


def lds_smoother(lds, inputs, data):
    """
    Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states.

    Args:

    rng:        jax.random.PRNGKey
    lds:        an LDS-like object (e.g. ssm_jax.lglds.models.LDS)
    inputs:     array of inputs to the LDS
    data:       array of data

    Returns:

    ll:         marginal log likelihood of the data
    Exs:        smoothed mean of the latent states.
    ExxTs:      smoothed second moments of the latent states.
    ExxnTs:     smoothed second moments of the latent states.
    """
    raise NotImplementedError
