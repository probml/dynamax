# Code for parameter estimation (MLE, MAP) using EM and SGD

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit
from jax import lax
from jax import value_and_grad
from jax import vmap
from tqdm.auto import trange


def hmm_fit_em(hmm, batch_emissions, optimizer=optax.adam(1e-2), num_iters=50):

    @jit
    def em_step(hmm):
        batch_posteriors, batch_trans_probs = hmm.e_step(batch_emissions)
        hmm, marginal_logliks = hmm.m_step(batch_emissions, batch_posteriors, batch_trans_probs, optimizer)
        return hmm, marginal_logliks

    log_probs = []
    for _ in trange(num_iters):
        hmm, marginal_logliks = em_step(hmm)
        log_probs.append(marginal_logliks[-1])

    return hmm, log_probs


def _loss_fn(hmm, batch_emissions, params):
    cls = hmm.__class__
    hypers = hmm.hyperparams
    hmm = cls.from_unconstrained_params(params, hypers)
    f = lambda emissions: -hmm.marginal_log_prob(emissions) / len(emissions)
    return vmap(f)(batch_emissions).mean()


def hmm_fit_sgd(hmm, batch_emissions, optimizer, num_iters=50, loss_fn=None):
    cls = hmm.__class__
    hypers = hmm.hyperparams

    if loss_fn is None:
        loss_fn = partial(_loss_fn, hmm, batch_emissions)

    loss_grad_fn = value_and_grad(loss_fn)

    @jit
    def opt_step(params, opt_state):
        val, grads = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, opt_state

    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)
    losses = []
    pbar = trange(num_iters)

    for step in pbar:
        loss_val, params, opt_state = opt_step(params, opt_state)
        losses.append(loss_val)
        # pbar.set_description("Loss={:.1f}".format(loss_val))

    hmm = cls.from_unconstrained_params(params, hypers)
    return hmm, jnp.stack(losses)


def _sample_minibatches(sequences, batch_size):
    n_seq = len(sequences)
    for idx in range(0, n_seq, batch_size):
        yield sequences[idx:min(idx + batch_size, n_seq)]


def hmm_fit_minibatch_gradient_descent(hmm, emissions, optimizer, batch_size=1, num_iters=50, key=jr.PRNGKey(0)):
    cls = hmm.__class__
    hypers = hmm.hyperparams

    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)

    num_complete_batches, leftover = jnp.divmod(len(emissions), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

    def loss(params, batch_emissions):
        hmm = cls.from_unconstrained_params(params, hypers)
        f = lambda emissions: -hmm.marginal_log_prob(emissions) / len(emissions)
        return vmap(f)(batch_emissions).mean()

    loss_grad_fn = jit(value_and_grad(loss))

    def train_step(carry, key):
        perm = jr.permutation(key, len(emissions))
        _emissions = emissions[perm]
        sample_generator = _sample_minibatches(_emissions, batch_size)

        def opt_step(carry, i):
            params, opt_state = carry
            batch = next(sample_generator)
            val, grads = loss_grad_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), val

        state, losses = lax.scan(opt_step, carry, jnp.arange(num_batches))
        return state, losses.mean()

    keys = jr.split(key, num_iters)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)

    losses = losses.flatten()
    hmm = cls.from_unconstrained_params(params, hypers)

    return hmm, losses
