# Code for parameter estimation (MLE, MAP) using EM and SGD

from functools import partial

import jax.numpy as jnp
import optax
from jax import jit
from jax import lax
from jax import value_and_grad
from jax import vmap
from tqdm.auto import trange


# Helper function to access parameters
def _get_params(x, dim, t):
    return x[t] if x.ndim == dim + 1 else x


def _compute_sum_transition_probs(transition_matrix, hmm_posterior):
    """Compute the transition probabilities from the HMM posterior messages.

    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """

    def _step(carry, args):
        filtered_probs, smoothed_probs_next, predicted_probs_next, t = args

        # Get parameters for time t
        A = _get_params(transition_matrix, 2, t)

        # Compute smoothed transition probabilities (Eq. 8.4 of Saarka, 2013)
        relative_probs_next = smoothed_probs_next / predicted_probs_next
        smoothed_trans_probs = filtered_probs[:, None] * A * relative_probs_next[None, :]
        smoothed_trans_probs /= smoothed_trans_probs.sum()
        return carry + smoothed_trans_probs, None

    # Initialize the recursion
    num_states = transition_matrix.shape[-1]
    num_timesteps = len(hmm_posterior.filtered_probs)
    sum_transition_probs, _ = lax.scan(_step, jnp.zeros((num_states, num_states)),
                                       (hmm_posterior.filtered_probs[:-1], hmm_posterior.smoothed_probs[1:],
                                        hmm_posterior.predicted_probs[1:], jnp.arange(num_timesteps - 1)))
    return sum_transition_probs


def _compute_all_transition_probs(transition_matrix, hmm_posterior):
    """Compute the transition probabilities from the HMM posterior messages.

    Args:
        transition_matrix (_type_): _description_
        hmm_posterior (_type_): _description_
    """
    filtered_probs = hmm_posterior.filtered_probs[:-1]
    smoothed_probs_next = hmm_posterior.smoothed_probs[1:]
    predicted_probs_next = hmm_posterior.predicted_probs[1:]
    relative_probs_next = smoothed_probs_next / predicted_probs_next
    transition_probs = filtered_probs[:, :, None] * \
        transition_matrix * relative_probs_next[:, None, :]
    return transition_probs


def compute_transition_probs(transition_matrix, hmm_posterior, reduce_sum=True):
    """Computer the posterior marginal distributions over (hid(t), hid(t+1)),
    ..math:
        q_{tij} = Pr(z_t=i, z_{t+1}=j | obs_{1:T})  for t=1,...,T-1

    If `reduce_sum` is True, return :math:`\sum_t q_{tij}`.

    Args:
        transition_matrix (array): the transition matrix
        hmm_posterior (HMMPosterior): Output of `hmm_smoother` or `hmm_two_filter_smoother`
        reduce_sum (bool, optional): Whether or not to return the
            sum of transition probabilities over time. Defaults to True, which is
            more memory efficient.

    Returns:
        array of transition probabilities. The shape is (num_states, num_states) if
            reduce_sum==True, otherwise (num_timesteps, num_states, num_states).
    """
    if reduce_sum:
        return _compute_sum_transition_probs(transition_matrix, hmm_posterior)
    else:
        return _compute_all_transition_probs(transition_matrix, hmm_posterior)


def hmm_fit_em(hmm, batch_emissions, optimizer=optax.adam(1e-2), num_iters=50):

    @jit
    def em_step(hmm):
        batch_posteriors, batch_trans_probs = hmm.e_step(batch_emissions)
        hmm, marginal_logliks = hmm.m_step(batch_emissions, batch_posteriors, batch_trans_probs, optimizer)
        return hmm, marginal_logliks[-1]

    log_probs = []
    for _ in trange(num_iters):
        hmm, marginal_loglik = em_step(hmm)
        log_probs.append(marginal_loglik)

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
