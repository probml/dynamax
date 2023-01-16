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
    logvar: PyTree[Float]
    

def init_bbb_params(key, model, batch_init):
    key_mean, key_logvar = jax.random.split(key)
    
    params_mean = model.init(key_mean, batch_init)
    flat_params, reconstruct_fn = ravel_pytree(params_mean)
    num_params = len(flat_params)
    
    params_logvar = jax.random.normal(key_logvar, (num_params,))
    params_logvar = reconstruct_fn(params_logvar)
    
    bbb_params = BBBParams(
        mean=params_mean,
        logvar=params_logvar,
    )
    
    return bbb_params, (reconstruct_fn, num_params)


def transform(eps, mean, logvar):
    std = jnp.exp(logvar / 2)
    weight = mean + std * eps
    return weight


@partial(jax.jit, static_argnames=("num_params", "reconstruct_fn"))
def sample_params(key, state:BBBParams, num_params, reconstruct_fn:Callable):
    eps = jax.random.normal(key, (num_params,))
    eps = reconstruct_fn(eps)
    params = jax.tree_map(transform, eps, state.mean, state.logvar)
    return params


@jax.jit
def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params


def cost_fn(
    key: jax.random.PRNGKey,
    state: BBBParams,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    reconstruct_fn: Callable,
    scale_obs=1.0,
    scale_prior=1.0,
):
    """
    TODO:
    Add more general way to compute observation-model log-probability
    """
    
    # Sampled params
    params = sample_params(key, state, num_params, reconstruct_fn)
    params_flat = get_leaves(params)
    
    # Prior log probability (use initialised vals for mean?)
    logp_prior = distrax.Normal(loc=0.0, scale=scale_prior).log_prob(params_flat).sum()
    # Observation log-probability
    mu_obs = model.apply(params, X).ravel()
    logp_obs = distrax.Normal(loc=mu_obs, scale=scale_obs).log_prob(y).sum()
    # Variational log-probability
    logp_variational = jax.tree_map(
        lambda mean, logvar, x: distrax.Normal(loc=mean, scale=jnp.exp(logvar / 2)).log_prob(x),
        state.mean, state.logvar, params
    )
    logp_variational = get_leaves(logp_variational).sum()
    
    return logp_variational - logp_prior - logp_obs


def lossfn(key, params, X, y, reconstruct_fn, num_samples=10):
    # TODO: add costfn as input
    keys = jax.random.split(key, num_samples)
    cost_vmap = jax.vmap(cost_fn, in_axes=(0, None, None, None, None))
    loss = cost_vmap(keys, params, X, y, reconstruct_fn).mean()
    return loss


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


@jax.jit
def index_values_batch(X, y, ixs):
    """
    Index values of a batch of observations
    """
    X_batch = X[ixs]
    y_batch = y[ixs]
    return X_batch, y_batch


@partial(jax.jit, static_argnames=("lossfn", "reconstruct_fn"))
def train_step(key, opt_state, X, y, lossfn, reconstruct_fn):
    params = opt_state.params
    apply_fn = opt_state.apply_fn
    
    loss, grads = jax.value_and_grad(lossfn, 1)(key, params, X, y, reconstruct_fn)
    opt_state_new = opt_state.apply_gradients(grads=grads)
    return opt_state_new, loss


def train_epoch(key, state, X, y, batch_size, lossfn, reconstruct_fn):
    num_samples = len(X)
    key_batch, keys_train = jax.random.split(key)
    batch_ixs = get_batch_train_ixs(key_batch, num_samples, batch_size)
    
    num_batches = len(batch_ixs)
    keys_train = jax.random.split(keys_train, num_batches)
    
    total_loss = 0
    for key_step, batch_ix in zip(keys_train, batch_ixs):
        X_batch, y_batch = index_values_batch(X, y, batch_ix)
        state, loss = train_step(key_step, state, X_batch, y_batch, lossfn, reconstruct_fn)
        total_loss += loss
    
    return total_loss.item(), state