"""
[2] Farquhar, S., Osborne, M., & Gal, Y. (2019).
    Radial Bayesian Neural Networks: Beyond Discrete Support
    In Large-Scale Bayesian Deep Learning. doi:10.48550/ARXIV.1907.00865

"""

import jax
import distrax
import jax.numpy as jnp
import flax.linen as nn
from chex import dataclass
from functools import partial
from typing import Callable
from jax.flatten_util import ravel_pytree
from jaxtyping import Float, PyTree, Array


@dataclass
class BBBParams:
    mean: PyTree[Float]
    rho: PyTree[Float]
    

def init_bbb_params(key, model, batch_init):
    key_mean, key_rho = jax.random.split(key)
    
    params_mean = model.init(key_mean, batch_init)
    flat_params, reconstruct_fn = ravel_pytree(params_mean)
    num_params = len(flat_params)

    params_rho = jax.random.normal(key_rho, (num_params,))
    params_rho = reconstruct_fn(params_rho)
     #
    bbb_params = BBBParams(
        mean=params_mean,
        rho=params_rho,
    )
    
    return bbb_params, (reconstruct_fn, num_params)


def transform(eps, mean, rho):
    std = jnp.log(1 + jnp.exp(rho))
    weight = mean + std * eps
    return weight


def sample_gauss_params(key, state:BBBParams, reconstruct_fn:Callable):
    """
    Sample from a Gaussian distribution
    """
    num_params = len(get_leaves(state.mean))
    eps = jax.random.normal(key, (num_params,))
    eps = reconstruct_fn(eps)

    params = jax.tree_map(transform, eps, state.mean, state.rho)
    return params


def sample_rbnn_params(key, state:BBBParams, reconstruct_fn:Callable, scale:float=1.0):
    """
    Sample from a radial Bayesian neural network
    radial BNN of [2]. We modify the definition of the
    RBNN to include a scale parameter, which allows us
    to control the prior uncertainty over the posterior predictive.
    """
    key_eps, key_rho = jax.random.split(key)
    num_params = len(get_leaves(state.mean))

    # The radial dimension.
    r = jax.random.normal(key_rho) * scale

    eps = jax.random.normal(key_eps, (num_params,))
    eps = eps / jnp.linalg.norm(eps) * r
    eps = reconstruct_fn(eps)


    params = jax.tree_map(transform, eps, state.mean, state.rho)
    return params


def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params


@partial(jax.jit, static_argnames=("num_samples", "batch_size"))
def get_batch_train_ixs(key, num_samples, batch_size):
    """
    Obtain the training indices to be used in an epoch of
    mini-batch optimisation.
    """
    steps_per_epoch = num_samples // batch_size
    
    batch_ixs = jax.random.permutation(key, num_samples)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    
    return batch_ixs


def index_values_batch(X, y, ixs):
    """
    Index values of a batch of observations
    """
    X_batch = X[ixs]
    y_batch = y[ixs]
    return X_batch, y_batch


def train_step(key, opt_state, X, y, lossfn, model, reconstruct_fn):
    params = opt_state.params
    apply_fn = opt_state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, X, y, model, reconstruct_fn)
    opt_state_new = opt_state.apply_gradients(grads=grads)
    return opt_state_new, loss


@partial(jax.jit, static_argnames=("lossfn", "model", "reconstruct_fn"))
def split_and_train_step(key, opt_state, X, y, ixs, lossfn, model, reconstruct_fn):
    X_batch, y_batch = index_values_batch(X, y, ixs)
    opt_state, loss = train_step(key, opt_state, X_batch, y_batch, lossfn, model, reconstruct_fn)
    return opt_state, loss


def train_epoch(key, state, X, y, batch_size, lossfn, model, reconstruct_fn):
    num_samples = len(X)
    key_batch, keys_train = jax.random.split(key)
    batch_ixs = get_batch_train_ixs(key_batch, num_samples, batch_size)
    
    num_batches = len(batch_ixs)
    keys_train = jax.random.split(keys_train, num_batches)
    
    total_loss = 0
    for key_step, batch_ix in zip(keys_train, batch_ixs):
        state, loss = split_and_train_step(key_step, state, X, y, batch_ix, lossfn, model, reconstruct_fn)
        total_loss += loss
    
    return total_loss.item(), state
