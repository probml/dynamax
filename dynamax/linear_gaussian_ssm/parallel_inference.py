"""
Parallel filtering and smoothing for a linear Gaussian SSM.

The trick is to write the filtering and smoothing computations in 
terms of a binary associative scan. This allows us to parallelize
the computation of the filtering and smoothing messages.

The filtering messages represent two distributions:
    
    f_ij(x_i, x_j) = p(x_i | y_{j+1:i}, x_j)
    g_ij(x_j) = p(y_i | x_j)

We define a binary associative operator that takes two filtering messages
and returns a new filtering message,

    (f_ij, g_ij) \otimes (f_jk, g_jk) = (f_ik, g_ik)

For a linear Gaussian SSM, the filtering messages are Gaussian potentials.
Following Sarkka and Garcia-Fernandez (2021), we write them in the following
form,
    
    f_{ij}(x_i, x_j) = N(x_i | A_{ij} x_j + b_{ij}, C_{ij})
    g_{ij}(x_j) \propto N_I(x_j | \eta_{ij}, J_{ij})

where A_{ij}, b_{ij}, C_{ij}, \eta_{ij}, J_{ij} are the parameters of the
Gaussian potentials. The updates for these parameters are derived in the 
manuscript referenced above.
"""
import jax.numpy as jnp
import warnings

from dynamax.utils.utils import symmetrize, psd_solve
from dynamax.linear_gaussian_ssm import PosteriorGSSMFiltered, PosteriorGSSMSmoothed, ParamsLGSSM
from dynamax.linear_gaussian_ssm.inference import _zeros_if_none
from dynamax.types import PRNGKeyT
from functools import partial
from jax import vmap, lax
from jax.scipy.linalg import cho_solve, cho_factor
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiagPlusLowRankCovariance as MVNLowRank,
    MultivariateNormalFullCovariance as MVN)
from typing import NamedTuple, Optional


def _get_one_param(x, dim, t):
    """Helper function to get one parameter at time t."""
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x

def _get_emission_cov_dim(params, num_timesteps):
    """Get the emission covariance dimension."""
    assert not callable(params.emissions.cov), "Emission covariance cannot be a callable for parallel inference."
    emission_dim = _get_one_param(params.emissions.bias, 1, 0).shape[0]
    R_shp = params.emissions.cov.shape
    if len(R_shp) == 1:
        assert R_shp[0] == emission_dim, "Emission covariance must have the same dimension as the emission bias."
        return 1
    
    elif len(R_shp) == 2: 
        if R_shp == (emission_dim, emission_dim):
            # Assume static, full covariance, but warn if it's TxT
            if emission_dim == num_timesteps:
                warnings.warn(
                    "Emission covariance has shape (T,T) where T is the number of timesteps. "
                    "The covariance will be interpreted as static and non-diagonal. To "
                    "specify a dynamic and diagonal covariance, pass it as a 3D array.")
            return 2
        elif R_shp == (num_timesteps, emission_dim):
            # Assume time-varying diagonal covariance
            return 1
        else:
            raise Exception("Emission covariance must be (T,D) or (D,D).")
        
    elif len(R_shp) == 3:
        # Time-varying full covariance
        assert R_shp == (num_timesteps, emission_dim, emission_dim)
        return 2
    else:
        raise Exception("Emission covariance must be a 2D or 3D array.")
            

#---------------------------------------------------------------------------#
#                                Filtering                                  #
#---------------------------------------------------------------------------#

def _emissions_scale(Q, H, R):
    """Compute the scale matrix for the emissions given the state covariance.

        S_inv = inv(H @ Q @ H.T + R)

    Args:
        Q (state_dim, state_dim): State covariance.
        H (emission_dim, state_dim): Emission matrix.
        R (emission_dim, emission_dim) or (emission_dim,): Emission covariance.

    Returns:
        K (state_dim, emission_dim): Kalman gain.
    """
    if R.ndim == 2:
        S = H @ Q @ H.T + R
        S_inv = psd_solve(S, jnp.eye(S.shape[0]))
    else:
        # Optimization using Woodbury identity with A=R, U=H@chol(Q), V=U.T, C=I
        # (see https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
        I = jnp.eye(Q.shape[0])
        U = H @ jnp.linalg.cholesky(Q)
        X = U / R[:, None]
        S_inv = jnp.diag(1.0 / R) - X @ psd_solve(I + U.T @ X, X.T) 
    return S_inv


def _marginal_loglik_elem(Q, H, R, y):
    """Compute marginal log-likelihood elements. 
    
    Args:
        Q (state_dim, state_dim): State covariance.
        H (emission_dim, state_dim): Emission matrix.
        R (emission_dim, emission_dim) or (emission_dim,): Emission covariance.
        y (emission_dim,): Emission.
    """
    if R.ndim == 2:
        S = H @ Q @ H.T + R
        return -MVN(jnp.zeros_like(y), S).log_prob(y)
    else:
        L = H @ jnp.linalg.cholesky(Q)
        return -MVNLowRank(jnp.zeros_like(y), R, L).log_prob(y)


class FilterMessage(NamedTuple):
    """
    Filtering associative scan elements.

    Attributes:
        A: P(z_j | y_{i:j}, z_{i-1}) weights.
        b: P(z_j | y_{i:j}, z_{i-1}) bias.
        C: P(z_j | y_{i:j}, z_{i-1}) covariance.
        J:   P(z_{i-1} | y_{i:j}) covariance.
        eta: P(z_{i-1} | y_{i:j}) mean.
        logZ: log P(y_{i:j}) marginal log-likelihood.
    """
    A:    Float[Array, "ntime state_dim state_dim"]
    b:    Float[Array, "ntime state_dim"]
    C:    Float[Array, "ntime state_dim state_dim"]
    J:    Float[Array, "ntime state_dim state_dim"]
    eta:  Float[Array, "ntime state_dim"]
    logZ: Float[Array, " ntime"]


def _initialize_filtering_messages(
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ):
    """Preprocess observations to construct input for filtering assocative scan."""

    num_timesteps = emissions.shape[0]
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))
    
    # Get the emission covariance dimension
    R_dim = _get_emission_cov_dim(params, num_timesteps)
    
    def _first_message(params, y, u):
        """Compute the first filtering message."""
        m = params.initial.mean
        P = params.initial.cov
        H = _get_one_param(params.emissions.weights, 2, 0)
        D = _get_one_param(params.emissions.input_weights, 2, 0)
        d = _get_one_param(params.emissions.bias, 1, 0)
        R = _get_one_param(params.emissions.cov, R_dim, 0)

        S = H @ P @ H.T + (R if R_dim == 2 else jnp.diag(R))
        S_inv = _emissions_scale(P, H, R)
        K = P @ H.T @ S_inv

        innov = y - D @ u - d
        A = jnp.zeros_like(P)
        b = m + K @ (innov - H @ m)
        C = symmetrize(P - K @ S @ K.T)
        eta = jnp.zeros_like(m)
        J = jnp.zeros_like(P)
        logZ = _marginal_loglik_elem(P, H, R, innov)
        return A, b, C, J, eta, logZ


    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _generic_message(params, y, ut, utm1, t):
        """Compute the generic filtering message."""
        F = _get_one_param(params.dynamics.weights, 2, t-1)
        B = _get_one_param(params.dynamics.input_weights, 2, t-1)
        b = _get_one_param(params.dynamics.bias, 1, t-1)
        Q = _get_one_param(params.dynamics.cov, 2, t-1)
        H = _get_one_param(params.emissions.weights, 2, t)
        D = _get_one_param(params.emissions.input_weights, 2, t)
        d = _get_one_param(params.emissions.bias, 1, t)
        R = _get_one_param(params.emissions.cov, R_dim, t)

        S_inv = _emissions_scale(Q, H, R)
        K = Q @ H.T @ S_inv
        
        bias_tm1 = B @ utm1 + b
        innov = (y - D @ ut - d - H @ bias_tm1)
        A = F - K @ H @ F
        b = bias_tm1 + K @ innov
        C = symmetrize(Q - K @ H @ Q)
        eta = F.T @ H.T @ S_inv @ innov
        J = symmetrize(F.T @ H.T @ S_inv @ H @ F)

        logZ = _marginal_loglik_elem(Q, H, R, innov)
        return A, b, C, J, eta, logZ

    A0, b0, C0, J0, eta0, logZ0 = _first_message(params, emissions[0], inputs[0])
    At, bt, Ct, Jt, etat, logZt = _generic_message(params, emissions[1:], inputs[1:], inputs[:-1], jnp.arange(1, len(emissions)))

    return FilterMessage(
        A=jnp.concatenate([A0[None], At]),
        b=jnp.concatenate([b0[None], bt]),
        C=jnp.concatenate([C0[None], Ct]),
        J=jnp.concatenate([J0[None], Jt]),
        eta=jnp.concatenate([eta0[None], etat]),
        logZ=jnp.concatenate([logZ0[None], logZt])
    )



def lgssm_filter(params: ParamsLGSSM,
                 emissions: Float[Array, "ntime emission_dim"],
                 inputs: Optional[Float[Array, "ntime input_dim"]]=None) \
                 -> PosteriorGSSMFiltered:
    """A parallel version of the lgssm filtering algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    """
    @vmap
    def _operator(elem1, elem2):
        """Parallel filtering operator."""
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

    initial_messages = _initialize_filtering_messages(params, emissions, inputs)
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
    E: Float[Array, "num_timesteps state_dim state_dim"]
    g: Float[Array, "num_timesteps state_dim"]
    L: Float[Array, "num_timesteps state_dim state_dim"]


def _initialize_smoothing_messages(params: ParamsLGSSM, 
                                   filtered_means: Float[Array, "num_timesteps state_dim"], 
                                   filtered_covariances: Float[Array, "num_timesteps state_dim state_dim"],
                                   inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
                                   ) -> SmoothMessage:
    """Preprocess filtering output to construct input for smoothing assocative scan."""

    def _last_message(m, P):
        """Compute the last smoothing message."""
        return jnp.zeros_like(P), m, P

    num_timesteps = filtered_means.shape[0]
    inputs = _zeros_if_none(inputs, (num_timesteps, 0))

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _generic_message(params, m, P, ut, t):
        """Compute the generic smoothing message."""
        F = _get_one_param(params.dynamics.weights, 2, t)
        B = _get_one_param(params.dynamics.input_weights, 2, t)
        b = _get_one_param(params.dynamics.bias, 1, t)
        Q = _get_one_param(params.dynamics.cov, 2, t)
        
        CF, low = cho_factor(F @ P @ F.T + Q)
        E = cho_solve((CF, low), F @ P).T
        g  = m - E @ (F @ m + B @ ut + b)
        L  = symmetrize(P - E @ F @ P)
        return E, g, L
    
    En, gn, Ln = _last_message(filtered_means[-1], filtered_covariances[-1])
    Et, gt, Lt = _generic_message(params, filtered_means[:-1], filtered_covariances[:-1], inputs[:-1], jnp.arange(len(filtered_means)-1))
    
    return SmoothMessage(
        E=jnp.concatenate([Et, En[None]]),
        g=jnp.concatenate([gt, gn[None]]),
        L=jnp.concatenate([Lt, Ln[None]])
    )


def lgssm_smoother(params: ParamsLGSSM,
                   emissions: Float[Array, "ntime emission_dim"],
                   inputs: Optional[Float[Array, "ntime input_dim"]]=None
                   ) -> PosteriorGSSMSmoothed:
    """A parallel version of the lgssm smoothing algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    """
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances
    
    @vmap
    def _operator(elem1, elem2):
        """Parallel smoothing operator."""
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2
        E = E2 @ E1
        g = E2 @ g1 + g2
        L = symmetrize(E2 @ L1 @ E2.T + L2)
        return E, g, L

    initial_messages = _initialize_smoothing_messages(params, filtered_means, filtered_covs, inputs)
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


def _initialize_sampling_messages(key, params, filtered_means, filtered_covariances,
                                  inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> SampleMessage:
    """A parallel version of the lgssm sampling algorithm.
    
    Given parallel smoothing messages `z_i ~ N(E_i z_{i+1} + g_i, L_i)`, 
    the parallel sampling messages are `(E_i,h_i)` where `h_i ~ N(g_i, L_i)`.
    """
    E, g, L = _initialize_smoothing_messages(params, filtered_means, filtered_covariances)
    return SampleMessage(E=E, h=MVN(g, L).sample(seed=key))


def lgssm_posterior_sample(
    key: PRNGKeyT,
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> Float[Array, "ntime state_dim"]:
    """A parallel version of the lgssm sampling algorithm.

    See S. Särkkä and Á. F. García-Fernández (2021) - https://arxiv.org/abs/1905.13002.
    """
    filtered_posterior = lgssm_filter(params, emissions, inputs)
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    @vmap
    def _operator(elem1, elem2):
        """Parallel sampling operator."""
        E1, h1 = elem1
        E2, h2 = elem2

        E = E2 @ E1
        h = E2 @ h1 + h2
        return E, h

    initial_messages = _initialize_sampling_messages(key, params, filtered_means, filtered_covs, inputs)
    _, samples = lax.associative_scan(_operator, initial_messages, reverse=True)
    return samples
