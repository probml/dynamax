# Online Bayesian linear regression in 1d using Kalman Filter
# Based on:
# https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m

# The latent state corresponds to the current estimate of the regression weights w.
#
# The observation model has the form:
#     p(y(t) |  w(t), x(t)) = N(y(t) | H(t) * w(t), R),
# where H(t) = X[t,...]is the observation matrix for step t.
#
# The dynamics model has the form:
#     p(w(t) | w(t-1)) = N( w(t) | F * w(t-1), Q),
# where Q>0 allows for parameter  drift.
#
# We show that the result is equivalent to batch (offline) Bayesian inference.
from jax import numpy as jnp
from matplotlib import pyplot as plt
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, LGSSMParams


def batch_bayesian_lreg(X, y, obs_var, mu0, Sigma0):
    """Compute posterior mean and covariance matrix of weights in Bayesian
    linear regression.

    The conditional probability of observations, y(t), given covariate x(t), and
    weights, w=[w_0, w_1] is given by:
         p(y(t) | x(t), w) = N(y(t) | w_0 + x(t) * w_1, obs_var),
    with a Gaussian prior over the weights:
        p(w) = N(w | mu0, Sigma0).

    Args:
        X: array(n_obs, dim) -  Matrix of features.
        y: array(n_obs,) - Array of observations.
        obs_var: float - Conditional variance of observations.
        mu0: array(dim) - Prior mean.
        Sigma0: array(dimesion, dim) Prior covariance matrix.
    Returns:
        * array(dim) - Posterior mean.
        * array(n_obs, dim, dim) - Posterior precision matrix.
    """
    posterior_prec = jnp.linalg.inv(Sigma0) + X.T @ X / obs_var
    b = jnp.linalg.inv(Sigma0) @ mu0 + X.T @ y / obs_var
    posterior_mean = jnp.linalg.solve(posterior_prec, b)

    return posterior_mean, posterior_prec


def kf_linreg(X, y, R, mu0, Sigma0, F, Q):
    """Online estimation of a linear regression using Kalman filtering.

    The latent state corresponds to the current estimate of the regression weights, w.

    The observation model has the form:
        p(y(t) |  w(t), x(t)) = N(y(t) | H(t) * w(t), R),
    where H(t) = X[t,...] is the emission matrix for step t and * is matrix-vector
    multiplication.

    The dynamics model has the form:
        p(w(t) | w(t-1)) = N(w(t) | F * w(t-1), Q),
    where Q>0 allows for parameter drift.

    Args:
        X: array(n_obs, 1, dim) -  Matrix of features, acts here as a
            non-stationary emission matrix with each row corresponding to the
            emission matrix, H(t) shape (1, dim), for an individual observation.
        y: array(n_obs, 1) - Array of observations.
        R: array(1, 1) Emission covariance matrix.
            The value of the single element is equal to the conditional variance
            of observations in the linear regression model.
        mu0: array(dim) - Prior mean.
        Sigma0: array(dimesion, dim) Prior covariance matrix.
        F: array(dim, dim) - lds dynamics matrix.
        Q: array(dim, dim) - lds dynamics covariance.

    Returns:
        * array(n_obs, dim) - Online estimates of posterior mean.
        * array(n_obs, dim, dim) - Online estimate of posterior covariance matrix.
    """
    lgssm = LGSSMParams(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_matrix=F,
        dynamics_covariance=Q,
        emission_matrix=X,
        emission_covariance=R,
    )

    lgssm_posterior = lgssm_filter(lgssm, y)
    return lgssm_posterior.filtered_means, lgssm_posterior.filtered_covariances


def online_kf_vs_batch_linreg():
    """Compare online linear regression with Kalman filtering vs batch solution."""

    n_obs = 21
    x = jnp.linspace(0, 20, n_obs)
    X = jnp.column_stack((jnp.ones_like(x), x))  # Design matrix.
    F = jnp.eye(2)
    Q = jnp.zeros((2, 2))  # No parameter drift.
    obs_var = 1.0
    R = jnp.ones((1, 1)) * obs_var
    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 10.0

    # Data from original matlab example
    y = jnp.array(
        [
            2.4865,
            -0.3033,
            -4.0531,
            -4.3359,
            -6.1742,
            -5.604,
            -3.5069,
            -2.3257,
            -4.6377,
            -0.2327,
            -1.9858,
            1.0284,
            -2.264,
            -0.4508,
            1.1672,
            6.6524,
            4.1452,
            5.2677,
            6.3403,
            9.6264,
            14.7842,
        ]
    )

    kf_results = kf_linreg(X[:, None, :], y[:, None], R, mu0, Sigma0, F, Q)
    batch_results = batch_bayesian_lreg(X, y, obs_var, mu0, Sigma0)

    return kf_results, batch_results


def plot_online_kf_vs_batch_linreg(kf_results, batch_results):
    """Plot a comparison of the online and batch results."""
    # Unpack kalman filter results
    post_weights_kf, post_sigma_kf = kf_results
    w0_kf_hist, w1_kf_hist = post_weights_kf.T
    w0_kf_err, w1_kf_err = jnp.sqrt(post_sigma_kf[:, [0, 1], [0, 1]].T)

    # Unpack batch results
    post_weights_batch, post_prec_batch = batch_results
    w0_post_batch, w1_post_batch = post_weights_batch
    Sigma_post_batch = jnp.linalg.inv(post_prec_batch)
    w0_std_batch, w1_std_batch = jnp.sqrt(Sigma_post_batch[[0, 1], [0, 1]])

    fig, ax = plt.subplots()
    timesteps = jnp.arange(len(w0_kf_hist))

    # Plot online kalman filter posterior.
    ax.errorbar(timesteps, w0_kf_hist, w0_kf_err, fmt="-o", label="$w_0$", color="black", fillstyle="none")
    ax.errorbar(timesteps, w1_kf_hist, w1_kf_err, fmt="-o", label="$w_1$", color="tab:red")

    # Plot batch posterior.
    ax.hlines(y=w0_post_batch, xmin=timesteps[0], xmax=timesteps[-1], color="black", label="$w_0$ batch")
    ax.hlines(
        y=w1_post_batch, xmin=timesteps[0], xmax=timesteps[-1], color="tab:red", linestyle="--", label="$w_1$ batch"
    )
    ax.fill_between(timesteps, w0_post_batch - w0_std_batch, w0_post_batch + w0_std_batch, color="black", alpha=0.4)
    ax.fill_between(timesteps, w1_post_batch - w1_std_batch, w1_post_batch + w1_std_batch, color="tab:red", alpha=0.4)

    ax.set_xlabel("time")
    ax.set_ylabel("weights")
    ax.legend()

    dict_figures = {"linreg_online_kalman": fig}
    return dict_figures


def main(test_mode=False):
    kf_results, batch_results = online_kf_vs_batch_linreg()
    if not test_mode:
        dict_figures = plot_online_kf_vs_batch_linreg(kf_results, batch_results)
        plt.show()


if __name__ == "__main__":
    main()
