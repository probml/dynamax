"""
External implementations of nlgssm algorithms to use for unit test.
Taken from https://github.com/petergchang/sarkka-jax
Based on Simo Särkkä (2013), “Bayesian Filtering and Smoothing,”
Available: https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
"""

import jax.numpy as jnp
from jax import vmap
from jax import lax
from jax import jacfwd


def ekf(m_0, P_0, f, Q, h, R, Y):
    """
    First-order additive EKF (Sarkka Algorithm 5.4)
    """
    num_timesteps = len(Y)
    # Compute Jacobians
    F, H = jacfwd(f), jacfwd(h)

    def _step(carry, t):
        """One step of EKF"""
        m_k, P_k = carry

        # Update
        v = Y[t] - h(m_k)
        S = jnp.atleast_2d(H(m_k) @ P_k @ H(m_k).T + R)
        K = P_k @ H(m_k).T @ jnp.linalg.inv(S)
        m_post = m_k + K @ v
        P_post = P_k - K @ S @ K.T

        # Prediction step
        m_pred = f(m_post)
        P_pred = F(m_post) @ P_post @ F(m_post).T + Q

        return (m_pred, P_pred), (m_post, P_post)

    carry = (m_0, P_0)
    _, (ms, Ps) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return ms, Ps


def eks(m_0, P_0, f, Q, h, R, Y):
    """
    First-order additive EK smoother
    """
    num_timesteps = len(Y)

    # Run ekf
    m_post, P_post = ekf(m_0, P_0, f, Q, h, R, Y)

    # Compute Jacobians
    F, H = jacfwd(f), jacfwd(h)

    def _step(carry, t):
        """One step of EKS"""
        m_k, P_k = carry

        # Prediction step
        m_pred = f(m_post[t])
        P_pred = F(m_post[t]) @ P_post[t] @ F(m_post[t]).T + Q
        G = P_post[t] @ F(m_post[t]).T @ jnp.linalg.inv(P_pred)

        # Update step
        m_sm = m_post[t] + G @ (m_k - m_pred)
        P_sm = P_post[t] + G @ (P_k - P_pred) @ G.T

        return (m_sm, P_sm), (m_sm, P_sm)

    carry = (m_post[-1], P_post[-1])
    _, (m_sm, P_sm) = lax.scan(_step, carry, jnp.arange(num_timesteps - 1), reverse=True)
    m_sm = jnp.concatenate((m_sm, jnp.array([m_post[-1]])))
    P_sm = jnp.concatenate((P_sm, jnp.array([P_post[-1]])))

    return m_sm, P_sm


def slf_additive(m_0, P_0, f, Q, h, R, Ef, Efdx, Eh, Ehdx, Y):
    """
    Additive SLF with closed-form expectations (Sarkka Algorithm 5.10)
    """
    num_timesteps = len(Y)

    def _step(carry, t):
        """One step of SLF"""
        m_k, P_k = carry

        # Update step
        v = Y[t] - Eh(m_k, P_k)
        S = jnp.atleast_2d(Ehdx(m_k, P_k) @ jnp.linalg.inv(P_k) @ Ehdx(m_k, P_k).T + R)
        K = Ehdx(m_k, P_k).T @ jnp.linalg.inv(S)
        m_post = m_k + K @ v
        P_post = P_k - K @ S @ K.T

        # Prediction step
        m_pred = Ef(m_post, P_post)
        P_pred = Efdx(m_post, P_post) @ jnp.linalg.inv(P_post) @ Efdx(m_post, P_post).T + Q

        return (m_pred, P_pred), (m_post, P_post)

    carry = (m_0, P_0)
    _, (ms, Ps) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return ms, Ps


def ukf(m_0, P_0, f, Q, h, R, alpha, beta, kappa, Y):
    """
    Additive UKF (Sarkka Algorithm 5.14)
    """
    num_timesteps, n = len(Y), P_0.shape[0]
    lamb = alpha**2 * (n + kappa) - n

    # Compute weights for mean and covariance estimates
    def compute_weights(n, alpha, beta, lamb):
        """Compute weights for UKF"""
        factor = 1 / (2 * (n + lamb))
        w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
        w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * n) * factor))
        return w_mean, w_cov

    w_mean, w_cov = compute_weights(n, alpha, beta, lamb)

    def _step(carry, t):
        """One step of UKF"""
        m_k, P_k = carry

        # Update step:
        # 1. Form sigma points
        sigmas_update = compute_sigmas(m_k, P_k, n, lamb)
        # 2. Propagate the sigma points
        sigmas_update_prop = vmap(h, 0, 0)(sigmas_update)
        # 3. Compute params
        mu = jnp.tensordot(w_mean, sigmas_update_prop, axes=1)
        outer = lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y)
        outer = vmap(outer, 0, 0)
        S = jnp.tensordot(w_cov, outer(sigmas_update_prop - mu, sigmas_update_prop - mu), axes=1) + R
        C = jnp.tensordot(w_cov, outer(sigmas_update - m_k, sigmas_update_prop - mu), axes=1)
        # 4. Compute posterior
        K = C @ jnp.linalg.inv(S)
        m_post = m_k + K @ (Y[t] - mu)
        P_post = P_k - K @ S @ K.T

        # Prediction step:
        # 1. Form sigma points
        sigmas_pred = compute_sigmas(m_post, P_post, n, lamb)
        # 2. Propagate the sigma points
        sigmas_pred = vmap(f, 0, 0)(sigmas_pred)
        # 3. Compute predicted mean and covariance
        m_pred = jnp.tensordot(w_mean, sigmas_pred, axes=1)
        P_pred = jnp.tensordot(w_cov, outer(sigmas_pred - m_pred, sigmas_pred - m_pred), axes=1) + Q

        return (m_pred, P_pred), (m_post, P_post)

    # Find 2n+1 sigma points
    def compute_sigmas(m, P, n, lamb):
        """Compute sigma points"""
        disc = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
        sigma_plus = jnp.array([m + disc[:, i] for i in range(n)])
        sigma_minus = jnp.array([m - disc[:, i] for i in range(n)])
        return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))

    carry = (m_0, P_0)
    _, (ms, Ps) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return ms, Ps


def uks(m_0, P_0, f, Q, h, R, alpha, beta, kappa, Y):
    """
    First-order additive UKS
    """
    num_timesteps, n = len(Y), P_0.shape[0]
    lamb = alpha**2 * (n + kappa) - n

    # Compute weights for mean and covariance estimates
    def compute_weights(n, alpha, beta, lamb):
        """Compute weights for UKS"""
        factor = 1 / (2 * (n + lamb))
        w_mean = jnp.concatenate((jnp.array([lamb / (n + lamb)]), jnp.ones(2 * n) * factor))
        w_cov = jnp.concatenate((jnp.array([lamb / (n + lamb) + (1 - alpha**2 + beta)]), jnp.ones(2 * n) * factor))
        return w_mean, w_cov

    w_mean, w_cov = compute_weights(n, alpha, beta, lamb)

    # Run ukf
    m_post, P_post = ukf(m_0, P_0, f, Q, h, R, alpha, beta, kappa, Y)

    def _step(carry, t):
        """One step of UKS"""
        m_k, P_k = carry
        m_p, P_p = m_post[t], P_post[t]

        # Prediction step
        sigmas_pred = compute_sigmas(m_p, P_p, n, lamb)
        sigmas_pred_prop = vmap(f, 0, 0)(sigmas_pred)
        m_pred = jnp.tensordot(w_mean, sigmas_pred_prop, axes=1)
        outer = lambda x, y: jnp.atleast_2d(x).T @ jnp.atleast_2d(y)
        outer = vmap(outer, 0, 0)
        P_pred = jnp.tensordot(w_cov, outer(sigmas_pred_prop - m_pred, sigmas_pred_prop - m_pred), axes=1) + Q
        P_cross = jnp.tensordot(w_cov, outer(sigmas_pred - m_p, sigmas_pred_prop - m_pred), axes=1)
        G = P_cross @ jnp.linalg.inv(P_pred)

        # Update step
        m_sm = m_p + G @ (m_k - m_pred)
        P_sm = P_p + G @ (P_k - P_pred) @ G.T

        return (m_sm, P_sm), (m_sm, P_sm)

    # Find 2n+1 sigma points
    def compute_sigmas(m, P, n, lamb):
        """Compute sigma points"""
        disc = jnp.sqrt(n + lamb) * jnp.linalg.cholesky(P)
        sigma_plus = jnp.array([m + disc[:, i] for i in range(n)])
        sigma_minus = jnp.array([m - disc[:, i] for i in range(n)])
        return jnp.concatenate((jnp.array([m]), sigma_plus, sigma_minus))

    carry = (m_post[-1], P_post[-1])
    _, (m_sm, P_sm) = lax.scan(_step, carry, jnp.arange(num_timesteps - 1), reverse=True)
    m_sm = jnp.concatenate((m_sm, jnp.array([m_post[-1]])))
    P_sm = jnp.concatenate((P_sm, jnp.array([P_post[-1]])))

    return m_sm, P_sm
