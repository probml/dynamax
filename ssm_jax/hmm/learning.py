# Code for parameter estimation (MLE, MAP) using EM and SGD

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jit, value_and_grad

import chex
import optax

from tqdm.auto import trange

from ssm_jax.hmm.inference import hmm_smoother


def hmm_fit_em(hmm, emissions, niter):

    @jit
    def em_step(hmm, emissions):
        posterior = hmm.smoother(emissions)
        hmm = hmm.m_step(emissions, posterior)
        return hmm, posterior

    log_probs = []
    for _ in trange(niter):
        hmm, posterior = em_step(hmm, emissions)
        log_probs.append(posterior.marginal_log_lkhd)

    return hmm, log_probs


def hmm_fit_sgd(cls, hmm, emissions, optimizer, niter):

  def loss(params):
      hmm = cls.from_unconstrained_params(params, ())
      return -hmm.marginal_log_prob(emissions) / len(emissions)
  loss_grad_fn = value_and_grad(loss)

  @jit
  def opt_step(params, opt_state):
      val, grads = loss_grad_fn(params)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return val, params, opt_state
      
  params = hmm.unconstrained_params
  opt_state = optimizer.init(params)
  losses = []
  pbar = trange(niter)
  for step in pbar:
      loss_val, params, opt_state = opt_step(params, opt_state)
      losses.append(loss_val)
      pbar.set_description("Loss={:.1f}".format(loss_val))

  hmm = cls.from_unconstrained_params(params, ())
  return hmm, losses