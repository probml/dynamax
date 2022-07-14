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


def hmm_fit_em(hmm, batch_emissions, num_iters=50, **kwargs):
    @jit
    def em_step(hmm):
        batch_posteriors = hmm.e_step(batch_emissions)
        hmm = hmm.m_step(batch_emissions, batch_posteriors, **kwargs)
        return hmm, batch_posteriors

    log_probs = []
    for _ in trange(num_iters):
        hmm, batch_posteriors = em_step(hmm)
        log_probs.append(batch_posteriors.marginal_loglik.sum())

    return hmm, log_probs


def _loss_fn(hmm, params, batch_emissions, lens):
    """Default objective function."""
    cls = hmm.__class__
    hypers = hmm.hyperparams
    hmm = cls.from_unconstrained_params(params, hypers)
    f = lambda emissions, t: -hmm.marginal_log_prob(emissions) / t
    return vmap(f)(batch_emissions, lens).mean()


def _sample_minibatches(key, sequences, lens, batch_size, shuffle):
    """Sequence generator."""
    n_seq = len(sequences)
    perm = jnp.where(shuffle, jr.permutation(key, n_seq), jnp.arange(n_seq))
    _sequences = sequences[perm]
    _lens = lens[perm]

    for idx in range(0, n_seq, batch_size):
        yield _sequences[idx : min(idx + batch_size, n_seq)], _lens[idx : min(idx + batch_size, n_seq)]


def hmm_fit_sgd(
    hmm,
    batch_emissions,
    lens=None,
    optimizer=optax.adam(1e-3),
    batch_size=1,
    num_iters=50,
    loss_fn=None,
    shuffle=False,
    key=jr.PRNGKey(0),
):
    """
    Note that batch_emissions is initially of shape (N,T)
    where N is the number of independent sequences and
    T is the length of a sequence. Then, a random susbet with shape (B, T)
    of entire sequence, not time steps, is sampled at each step where B is
    batch size.

    Args:
        hmm (BaseHMM): HMM class whose parameters will be estimated.
        batch_emissions (chex.Array): Independent sequences.
        optmizer (optax.Optimizer): Optimizer.
        batch_size (int): Number of sequences used at each update step.
        num_iters (int): Iterations made on only one mini-batch.
        loss_fn (Callable): Objective function.
        shuffle (bool): Indicates whether to shuffle emissions.
        key (chex.PRNGKey): RNG key.

    Returns:
        hmm: HMM with optimized parameters.
        losses: Output of loss_fn stored at each step.
    """
    cls = hmm.__class__
    hypers = hmm.hyperparams

    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)

    if lens is None:
        num_sequences, num_timesteps = batch_emissions.shape
        lens = jnp.ones((num_sequences,)) * num_timesteps

    if batch_size == len(batch_emissions):
        shuffle = False

    num_complete_batches, leftover = jnp.divmod(len(batch_emissions), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

    if loss_fn is None:
        loss_fn = partial(_loss_fn, hmm)

    loss_grad_fn = value_and_grad(loss_fn)

    def train_step(carry, key):

        sample_generator = _sample_minibatches(key, batch_emissions, lens, batch_size, shuffle)

        def opt_step(carry, i):
            params, opt_state = carry
            batch, ts = next(sample_generator)
            val, grads = loss_grad_fn(params, batch, ts)
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
