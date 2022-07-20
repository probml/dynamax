# Code for parameter estimation (MLE, MAP) using EM and SGD

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax

from tqdm.auto import trange


def lgssm_fit_em(model, batch_emissions, num_iters=50):
    @jit
    def em_step(model):
        posterior_stats, marginal_loglikes = model.e_step(batch_emissions)
        model = model.m_step(posterior_stats)
        return model, marginal_loglikes.sum()

    log_probs = []
    for _ in trange(num_iters):
        model, marginal_loglik = em_step(model)
        log_probs.append(marginal_loglik)

    return model, jnp.array(log_probs)


# def hmm_fit_sgd(hmm, batch_emissions, optimizer, num_iters=50):
#     cls = hmm.__class__
#     hypers = hmm.hyperparams

#     def loss(params):
#         hmm = cls.from_unconstrained_params(params, hypers)
#         f = lambda emissions: -hmm.marginal_log_prob(emissions) / len(emissions)
#         return vmap(f)(batch_emissions).mean()
#     loss_grad_fn = value_and_grad(loss)

#     @jit
#     def opt_step(params, opt_state):
#         val, grads = loss_grad_fn(params)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return val, params, opt_state

#     params = hmm.unconstrained_params
#     opt_state = optimizer.init(params)
#     losses = []
#     pbar = trange(num_iters)
#     for step in pbar:
#         loss_val, params, opt_state = opt_step(params, opt_state)
#         losses.append(loss_val)
#         pbar.set_description("Loss={:.1f}".format(loss_val))

#     hmm = cls.from_unconstrained_params(params, hypers)
#     return hmm, jnp.stack(losses)
