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

    # params_rho = jax.random.normal(key_rho, (num_params,)) * 0.1
    # params_rho = reconstruct_fn(params_rho)

    params_rho = model.init(key_rho, batch_init)
    
    bbb_params = BBBParams(
        mean=params_mean,
        rho=params_rho,
    )
    
    return bbb_params, (reconstruct_fn, num_params)


def transform(eps, mean, rho):
    std = jnp.log(1 + jnp.exp(rho))
    weight = mean + std * eps
    return weight


def sample_params(key, state:BBBParams, reconstruct_fn:Callable):
    num_params = len(get_leaves(state.mean))
    eps = jax.random.normal(key, (num_params,))
    eps = reconstruct_fn(eps)
    params = jax.tree_map(transform, eps, state.mean, state.rho)
    return params


def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params


def cost_fn(
    key: jax.random.PRNGKey,
    state: BBBParams,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    reconstruct_fn: Callable,
    model: nn.Module,
    scale_obs=1.0,
    scale_prior=1.0,
):
    """
    TODO:
    Add more general way to compute observation-model log-probability
    """
    # Sampled params
    params = sample_params(key, state, reconstruct_fn)
    params_flat = get_leaves(params)
    
    # Prior log probability (use initialised vals for mean?)
    logp_prior = distrax.Normal(loc=0.0, scale=scale_prior).log_prob(params_flat).sum()
    # Observation log-probability
    mu_obs = model.apply(params, X).ravel()
    logp_obs = distrax.Normal(loc=mu_obs, scale=scale_obs).log_prob(y).sum()
    # Variational log-probability
    logp_variational = jax.tree_map(
        lambda mean, rho, x: distrax.Normal(loc=mean, scale=jnp.log(1 + jnp.exp(rho))).log_prob(x),
        state.mean, state.rho, params
    )
    logp_variational = get_leaves(logp_variational).sum()
    
    return logp_variational - logp_prior - logp_obs


def lossfn(key, params, X, y, model, reconstruct_fn, num_samples=10):
    # TODO: add costfn as input
    keys = jax.random.split(key, num_samples)
    cost_vmap = jax.vmap(cost_fn, in_axes=(0, None, None, None, None, None))
    loss = cost_vmap(keys, params, X, y, reconstruct_fn, model).mean()
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
