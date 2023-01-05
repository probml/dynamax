"""
Implementation of the Recursive Variational Gaussian Approximation
(R-VGA) and the Limited-memory Recursive Variational Gaussian Approximation
(LR-VGA) [1] algorithms for sequential estimation.

[1] Lambert, M., Bonnabel, S., & Bach, F. (2021, December).
The limited-memory recursive variational Gaussian approximation (L-RVGA).
Retrieved from https://hal.inria.fr/hal-03501920
"""

import jax
import chex
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable
from functools import partial
from dataclasses import dataclass
from jaxtyping import Array, Float
from jax.flatten_util import ravel_pytree

from flax import struct


@chex.dataclass
class LRVGAState:
    mu: Float[Array, "dim_params"]
    W: Float[Array, "dim_params dim_subspace"]
    Psi: Float[Array, "dim_params"]

@struct.dataclass
class FlaxLRVGAState:
    mu: Float[Array, "dim_params"]
    W: Float[Array, "dim_params dim_subspace"]
    Psi: Float[Array, "dim_params"]


@dataclass
class Config:
    """
    Static component of the LRVGA algorithm.

    Parameters
    ----------
    num_samples
        Number of samples to draw from the variational Gaussian approx
    dim_latent
        Dimensionality of the latent subspace
    model
        Flax model to use for the variational Gaussian approx
    reconstruct_fn
        Reconstructs the model parameters from the flattened vector
    """
    num_samples: int
    dim_latent: int
    model: nn.Module
    reconstruct_fn: Callable


def init_state_lrvga(key, model, X, dim_latent, sigma2_init, num_samples, eps):
    key_W, key_mu = jax.random.split(key)

    mu_init = model.init(key_mu, X)
    mu_init, reconstruct_fn = ravel_pytree(mu_init)
    mu_init = jnp.array(mu_init)
    dim_params = len(mu_init)

    psi0 = (1 - eps) / sigma2_init
    w0 = jnp.sqrt((eps * dim_params) / (dim_latent * sigma2_init))
    
    W_init = jax.random.normal(key_W, (dim_params, dim_latent))
    W_init = W_init / jnp.linalg.norm(W_init, axis=0) * w0
    Psi_init = jnp.ones(dim_params) * psi0
    
    state_init = FlaxLRVGAState(
        mu=mu_init,
        W=W_init,
        Psi=Psi_init,
    )

    config = Config(
        dim_latent=dim_latent,
        model=model,
        reconstruct_fn=reconstruct_fn,
        num_samples=num_samples,
    )
    
    return state_init, config


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
def sample_lrvga(key, state):
    """
    Sample from a low-rank variational Gaussian approximation.
    This implementation avoids the explicit construction of the
    (D x D) covariance matrix.

    We take s ~ N(0, W W^T + Psi I)

    Implementation based on §4.2.2 of the L-RVGA paper.

    TODO(?): refactor code into jax.vmap. (It faster?)
    """
    key_x, key_eps = jax.random.split(key)
    dim_full, dim_latent = state.W.shape
    Psi_inv = 1 / state.Psi

    eps_sample = jax.random.normal(key_eps, (dim_latent,))
    x_sample = jax.random.normal(key_x, (dim_full,)) * jnp.sqrt(Psi_inv)

    I_full = jnp.eye(dim_full)
    I_latent = jnp.eye(dim_latent)
    # M = I + W^T Psi^{-1} W
    M = I_latent + jnp.einsum("ji,j,jk->ik", state.W, Psi_inv, state.W)
    # L = Psi^{-1} W^T M^{-1}
    L_tr = jnp.linalg.solve(M.T, jnp.einsum("i,ij->ji", Psi_inv, state.W))

    # samples = (I - LW^T)x + Le
    term1 = I_full - jnp.einsum("ji,kj->ik", L_tr, state.W)
    x_transform = jnp.einsum("ij,j->i", term1, x_sample)
    eps_transform = jnp.einsum("ji,j->i", L_tr, eps_sample)
    samples = x_transform + eps_transform
    return samples + state.mu


def compute_lrvga_cov(state, x, y, link_fn, cov_fn):
    """
    Compute the approximated expected covariance matrix
    of the posterior distribution.

    Implementation based on §4.2.4 of the L-RVGA paper.
    """
    raise NotImplementedError("TODO")
    cov = cov_fn(state.mu, x, y)
    jax.vmap(jax.jvp, in_axes=(None, None, 0))(link_fn, (state.mu,), (cov,))
    # jax.vjp()


@partial(jax.vmap, in_axes=(0, None, None, None))
def sample_predictions(key, state, x, config):
    mu_sample = sample_lrvga(key, state)
    mu_sample = config.reconstruct_fn(mu_sample)
    yhat = config.model.apply(mu_sample, x, method=config.model.get_mean)
    return yhat


@partial(jax.vmap, in_axes=(0, None, None, None, None))
def sample_grad_expected_log_prob(key, state, x, y, config):
    """
    E[∇ logp(y|x,θ)]
    """
    mu_sample = sample_lrvga(key, state)
    mu_sample = config.reconstruct_fn(mu_sample)
    grad_log_prob = partial(config.model.apply, method=config.model.log_prob)
    grad_log_prob = jax.grad(grad_log_prob, argnums=0)
    grads = grad_log_prob(mu_sample, x, y)
    grads, _ = ravel_pytree(grads)
    return grads


def mu_update(
    key,
    x: Float[Array, "dim_obs"],
    y: float,
    state: LRVGAState,
    config: Config,
) -> Float[Array, "dim_obs"]:
    """
    TODO: Optimise for lower compilation time:
        1. Refactor sample_predictions
        2. Refactor sample_grad_expected_log_prob
    TODO: Rewrite the V term using the Woodbury matrix identity
    """
    mu = state.mu
    W = state.W
    Psi_inv = 1 / state.Psi
    dim_full, _ = W.shape
    I = jnp.eye(dim_full)

    keys = jax.random.split(key, config.num_samples)
    yhat = sample_predictions(keys, state, x, config).mean(axis=0)
    err = y - yhat

    V = W @ W.T - Psi_inv * I
    exp_grads_log_prob = sample_grad_expected_log_prob(keys, state, x, y, config).mean(axis=0)
    gain = jnp.linalg.solve(V, exp_grads_log_prob)
    mu_update = mu + gain * err
    return mu_update


def _step_lrvga(state, obs, alpha, beta, n_inner):
    x, y = obs

    # Xt = compute_lrvga_cov(state, x, y)
    # Then, replace x with Xt
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
