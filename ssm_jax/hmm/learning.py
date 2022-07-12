import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import optax
from tqdm.auto import trange

from ssm_jax.utils import sgd_helper


def hmm_fit_em(hmm, batch_emissions, num_iters=50, **batch_covariates):
    @jit
    def em_step(hmm):
        batch_posterior_stats, marginal_logliks = hmm.e_step(batch_emissions, **batch_covariates)
        hmm.m_step(batch_emissions, batch_posterior_stats, **batch_covariates)
        return hmm, marginal_logliks.sum()

    log_probs = []
    for _ in trange(num_iters):
        hmm, marginal_loglik = em_step(hmm)
        log_probs.append(marginal_loglik)

    return hmm, log_probs


def hmm_fit_sgd(
    hmm,
    batch_emissions,
    optimizer=optax.adam(1e-3),
    batch_size=1,
    num_iters=50,
    shuffle=False,
    key=jr.PRNGKey(0),
    **batch_covariates):
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

    def _loss_fn(params, batch_emissions, **batch_covariates):
        hmm = cls.from_unconstrained_params(params, hypers)
        f = lambda emissions, **covariates: \
            -hmm.marginal_log_prob(emissions, **covariates) / len(emissions)
        return vmap(f)(batch_emissions, **batch_covariates).mean()

    params, losses = sgd_helper(_loss_fn,
                                hmm.unconstrained_params,
                                batch_emissions,
                                optimizer=optimizer,
                                batch_size=batch_size,
                                num_iters=num_iters,
                                shuffle=shuffle,
                                key=key,
                                **batch_covariates)

    losses = losses.flatten()
    hmm = cls.from_unconstrained_params(params, hypers)
    return hmm, losses
