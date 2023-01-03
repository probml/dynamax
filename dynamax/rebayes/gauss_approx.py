"""
Implementation of the Recursive Variational Gaussian Approximation
(R-VGA) and the Limited-memory Recursive Variational Gaussian Approximation
(LR-VGA) algorithms for sequential estimation.
"""

import jax
import chex
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float


@chex.dataclass
class LRVGAState:
    mu: Float[Array, "dim_params"]
    W: Float[Array, "dim_params dim_subspace"]
    Psi: Float[Array, "dim_params"]


def init_state_lrvga(key, X, dim_latent, sigma2_init, eps):
    key_W, key_mu = jax.random.split(key)
    _, dim_obs = X.shape
    psi0 = (1 - eps) / sigma2_init
    w0 = jnp.sqrt((eps * dim_obs) / (dim_latent * sigma2_init))
    
    W_init = jax.random.normal(key_W, (dim_obs, dim_latent))
    W_init = W_init / jnp.linalg.norm(W_init, axis=0) * w0
    Psi_init = jnp.ones(dim_obs) * psi0
    
    # mu_init = jax.random.normal(key_mu, (dim_obs,))
    mu_init = jnp.zeros((dim_obs,))
    
    state_init = LRVGAState(
        mu=mu_init,
        W=W_init,
        Psi=Psi_init
    )
    
    return state_init


def fa_approx_step(
    x: Float[Array, "dim_params"],
    state: LRVGAState,
    state_prev: LRVGAState,
    alpha: float,
    beta: float
) -> LRVGAState:
    """
    Factor Analysis (FA) approximation to the low-rank (W)
    and diagonal (Psi) matrices.
    """
    # Load data
    W_prev, Psi_prev = state_prev.W, state_prev.Psi
    W, Psi = state.W, state.Psi
    
    # Initialise basic transformations
    dim_obs, dim_latent = W.shape
    I = jnp.eye(dim_latent)
    Psi_inv = 1 / Psi
    
    # Construct helper matrices
    M = I + jnp.einsum("ij,i,ik->jk", W, Psi_inv, W)
    M_inv = jnp.linalg.inv(M)
    V_beta = jnp.einsum("i,j,j,jk->ik", x, x, Psi_inv, W)
    V_alpha = (
        jnp.einsum("ij,kj,k,kl->il", W_prev, W_prev, Psi_inv, W) +
        jnp.einsum("i,i,ij->ij", Psi_prev, Psi_inv, W)
    )
    V = beta * V_beta + alpha * V_alpha
    # Value_update
    # (return transpose of W_solve -- avoid extra transpose op)
    W_solve = I + jnp.einsum("ij,kj,k,kl->li", M_inv, W, Psi_inv, V)
    W = jnp.linalg.solve(W_solve, V.T).T
    Psi = (
        beta * jnp.einsum("i,i->i", x, x) +
        alpha * jnp.einsum("ij,ij->i", W_prev, W_prev) + 
        alpha * Psi_prev -
        jnp.einsum("ij,jk,ik->i", W, M_inv, V)
    )
    
    new_state = state.replace(
        mu=state.mu,
        W=W,
        Psi=Psi
    )
    return new_state


@jax.jit
def mu_update(
    state: LRVGAState,
    x: Float[Array, "dim_obs"],
    y: float
) -> Float[Array, "dim_obs"]:
    mu = state.mu
    W = state.W
    Psi_inv = 1 / state.Psi
    _, dim_latent = W.shape
    I = jnp.eye(dim_latent)
    
    err = y - jnp.einsum("i,i->", mu, x)

    M = I + jnp.einsum("ij,i,ik->jk", W, Psi_inv, W)
    b_matrix = jnp.einsum("ij,i,i->j", W, Psi_inv, x)
    term_1 = jnp.linalg.solve(M, b_matrix)
    term_2 = jnp.einsum("ij,j->i", W, term_1)
    mu = mu + Psi_inv * (x - term_2) * err
    return mu


def _step_lrvga(state, obs, alpha, beta, n_inner):
    x, y = obs

    def fa_partial(_, new_state):
        new_state = fa_approx_step(x, new_state, state, alpha, beta)
        return new_state

    # Algorithm 1 in §3.2 of L-RVGA states that 1 to 3 loops may be enough in
    # the inner (fa-update) loop
    state_update = jax.lax.fori_loop(0, n_inner, fa_partial, state)
    mu_new = mu_update(state_update, x, y)
    state_update = state_update.replace(mu=mu_new)
    
    return state_update, mu_new


def lrvga(
    state_init: LRVGAState,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    alpha: float,
    beta: float,
    n_inner: int = 3,
):
    part_lrvga = partial(_step_lrvga, alpha=alpha, beta=beta, n_inner=n_inner)
    obs = (X, y)
    state_final, mu_hist = jax.lax.scan(part_lrvga, state_init, obs)
    return state_final, mu_hist
