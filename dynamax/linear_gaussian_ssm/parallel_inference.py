# Parallel filtering and smoothing for a lgssm.
# This implementation is adapted from the work of Adrien Correnflos in,
#  https://github.com/EEA-sensors/sequential-parallelization-examples/

import jax.numpy as jnp
from jax import vmap, lax
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import NamedTuple
from dynamax.types import PRNGKey
from functools import partial

from jax.scipy.linalg import cho_solve, cho_factor
from dynamax.utils.utils import symmetrize
from dynamax.linear_gaussian_ssm import PosteriorGSSMFiltered, PosteriorGSSMSmoothed, ParamsLGSSM


def _get_params(x, dim, t):
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x
    
#---------------------------------------------------------------------------#
#                                Filtering                                  #
#---------------------------------------------------------------------------#

class FilterMessage(NamedTuple):
    """
    Filtering associative scan elements.

    Attributes:
        A: P(z_j | y_{i:j}, z_{i-1}) weights.
        b: P(z_j | y_{i:j}, z_{i-1}) bias.
        C: P(z_j | y_{i:j}, z_{i-1}) covariance.
        J:   P(z_{i-1} | y_{i:j}) covariance.
        eta: P(z_{i-1} | y_{i:j}) mean.
    """
    A:    Float[Array, "ntime state_dim state_dim"]
    b:    Float[Array, "ntime state_dim"]
    C:    Float[Array, "ntime state_dim state_dim"]
    J:    Float[Array, "ntime state_dim state_dim"]
    eta:  Float[Array, "ntime state_dim"]
    logZ: Float[Array, "ntime"]


def _initialize_filtering_messages(params, emissions):
    """Preprocess observations to construct input for filtering assocative scan."""

    def _first_message(params, y):
        H = _get_params(params.emissions.weights, 2, 0)
        R = _get_params(params.emissions.cov, 2, 0)
        d = _get_params(params.emissions.bias, 1, 0)
        m = params.initial.mean
        P = params.initial.cov

        S = H @ P @ H.T + R
        CF, low = cho_factor(S)
        K = cho_solve((CF, low), H @ P).T

        A = jnp.zeros_like(P)
        b = m + K @ (y - H @ m - d)
        C = symmetrize(P - K @ S @ K.T)
        eta = jnp.zeros_like(b)
        J = jnp.eye(len(b))

        logZ = -MVN(loc=jnp.zeros_like(y), covariance_matrix=H @ P @ H.T + R).log_prob(y)
        return A, b, C, J, eta, logZ


    @partial(vmap, in_axes=(None, 0, 0))
    def _generic_message(params, y, t):
        F = _get_params(params.dynamics.weights, 2, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        H = _get_params(params.emissions.weights, 2, t+1)
        R = _get_params(params.emissions.cov, 2, t+1)
        d = _get_params(params.emissions.bias, 1, t+1)

        S = H @ Q @ H.T + R
        CF, low = cho_factor(S)
        K = cho_solve((CF, low), H @ Q).T

        eta = F.T @ H.T @ cho_solve((CF, low), y - H @ b - d)
        J = symmetrize(F.T @ H.T @ cho_solve((CF, low), H @ F))

        A = F - K @ H @ F
        b = b + K @ (y - H @ b - d)
        C = symmetrize(Q - K @ H @ Q)

        logZ = -MVN(loc=jnp.zeros_like(y), covariance_matrix=S).log_prob(y)
        return A, b, C, J, eta, logZ


    A0, b0, C0, J0, eta0, logZ0 = _first_message(params, emissions[0])
    At, bt, Ct, Jt, etat, logZt = _generic_message(params, emissions[1:], jnp.arange(len(emissions)-1))

    return FilterMessage(
        A=jnp.concatenate([A0[None], At]),
        b=jnp.concatenate([b0[None], bt]),
        C=jnp.concatenate([C0[None], Ct]),
        J=jnp.concatenate([J0[None], Jt]),
        eta=jnp.concatenate([eta0[None], etat]),
        logZ=jnp.concatenate([logZ0[None], logZt])
    )



def lgssm_filter(
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"]
) -> PosteriorGSSMFiltered:
    """A parallel version of the lgssm filtering algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.

    Note: This function does not yet handle `inputs` to the system.
    """
    @vmap
    def _operator(elem1, elem2):
        A1, b1, C1, J1, eta1, logZ1 = elem1
        A2, b2, C2, J2, eta2, logZ2 = elem2
        I = jnp.eye(A1.shape[0])

        I_C1J2 = I + C1 @ J2
        temp = jnp.linalg.solve(I_C1J2.T, A2.T).T
        A = temp @ A1
        b = temp @ (b1 + C1 @ eta2) + b2
        C = symmetrize(temp @ C1 @ A2.T + C2)

        I_J2C1 = I + J2 @ C1
        temp = jnp.linalg.solve(I_J2C1.T, A1).T
        eta = temp @ (eta2 - J2 @ b1) + eta1
        J = symmetrize(temp @ J2 @ A1 + J1)

        mu = jnp.linalg.solve(C1, b1)
        t1 = (b1 @ mu - (eta2 + mu) @ jnp.linalg.solve(I_C1J2, C1 @ eta2 + b1))
        logZ = (logZ1 + logZ2 + 0.5 * jnp.linalg.slogdet(I_C1J2)[1] + 0.5 * t1)
        return FilterMessage(A, b, C, J, eta, logZ)

    initial_messages = _initialize_filtering_messages(params, emissions)
    final_messages = lax.associative_scan(_operator, initial_messages)

    return PosteriorGSSMFiltered(
        filtered_means=final_messages.b,
        filtered_covariances=final_messages.C,
        marginal_loglik=-final_messages.logZ[-1])


#---------------------------------------------------------------------------#
#                                 Smoothing                                 #
#---------------------------------------------------------------------------#

class SmoothMessage(NamedTuple):
    """
    Smoothing associative scan elements.

    Attributes:
        E: P(z_i | y_{1:j}, z_{j+1}) weights.
        g: P(z_i | y_{1:j}, z_{j+1}) bias.
        L: P(z_i | y_{1:j}, z_{j+1}) covariance.
    """
    E: Float[Array, "ntime state_dim state_dim"]
    g: Float[Array, "ntime state_dim"]
    L: Float[Array, "ntime state_dim state_dim"]


def _initialize_smoothing_messages(params, filtered_means, filtered_covariances):
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    def _last_message(m, P):
        return jnp.zeros_like(P), m, P

    @partial(vmap, in_axes=(None, 0, 0, 0))
    def _generic_message(params, m, P, t):
        F = _get_params(params.dynamics.weights, 2, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)

        CF, low = cho_factor(F @ P @ F.T + Q)
        E = cho_solve((CF, low), F @ P).T
        g  = m - E @ (F @ m + b)
        L  = symmetrize(P - E @ F @ P)
        return E, g, L
    
    En, gn, Ln = _last_message(filtered_means[-1], filtered_covariances[-1])
    Et, gt, Lt = _generic_message(params, filtered_means[:-1], filtered_covariances[:-1], jnp.arange(len(filtered_means)-1))
    
    return SmoothMessage(
        E=jnp.concatenate([Et, En[None]]),
        g=jnp.concatenate([gt, gn[None]]),
        L=jnp.concatenate([Lt, Ln[None]])
    )


def lgssm_smoother(
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"]
) -> PosteriorGSSMSmoothed:
    """A parallel version of the lgssm smoothing algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.

    Note: This function does not yet handle `inputs` to the system.
    """
    filtered_posterior = lgssm_filter(params, emissions)
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances
    
    @vmap
    def _operator(elem1, elem2):
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2
        E = E2 @ E1
        g = E2 @ g1 + g2
        L = symmetrize(E2 @ L1 @ E2.T + L2)
        return E, g, L

    initial_messages = _initialize_smoothing_messages(params, filtered_means, filtered_covs)
    final_messages = lax.associative_scan(_operator, initial_messages, reverse=True)

    return PosteriorGSSMSmoothed(
        marginal_loglik=filtered_posterior.marginal_loglik,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=final_messages.g,
        smoothed_covariances=final_messages.L
    )


#---------------------------------------------------------------------------#
#                                 Sampling                                  #
#---------------------------------------------------------------------------#

class SampleMessage(NamedTuple):
    """
    Sampling associative scan elements.

    Attributes:
        E: z_i ~ z_{j+1} weights.
        h: z_i ~ z_{j+1} bias.
    """
    E: Float[Array, "ntime state_dim state_dim"]
    h: Float[Array, "ntime state_dim"]


def _initialize_sampling_messages(key, params, filtered_means, filtered_covariances):
    """A parallel version of the lgssm sampling algorithm.
    
    Given parallel smoothing messages `z_i ~ N(E_i z_{i+1} + g_i, L_i)`, 
    the parallel sampling messages are `(E_i,h_i)` where `h_i ~ N(g_i, L_i)`.
    """
    E, g, L = _initialize_smoothing_messages(params, filtered_means, filtered_covariances)
    return SampleMessage(E=E, h=MVN(g, L).sample(seed=key))


def lgssm_posterior_sample(
    key: PRNGKey,
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"]
) -> Float[Array, "ntime state_dim"]:
    """A parallel version of the lgssm sampling algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.

    Note: This function does not yet handle `inputs` to the system.
    """
    filtered_posterior = lgssm_filter(params, emissions)
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    @vmap
    def _operator(elem1, elem2):
        E1, h1 = elem1
        E2, h2 = elem2

        E = E2 @ E1
        h = E2 @ h1 + h2
        return E, h

    initial_messages = _initialize_sampling_messages(key, params, filtered_means, filtered_covs)
    _, samples = lax.associative_scan(_operator, initial_messages, reverse=True)
    return samples