# Code for parameter estimation (MLE, MAP) using EM and SGD

from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, lax, value_and_grad, tree_map
import optax

from tqdm.auto import trange

def hmm_fit_em(hmm, batch_emissions, num_iters=50, **kwargs):

    @jit
    def em_step(hmm):
        batch_posteriors = hmm.e_step(batch_emissions)
        hmm.m_step(batch_emissions, batch_posteriors, **kwargs)
        return hmm, batch_posteriors

    log_probs = []
    for _ in trange(num_iters):
        hmm, batch_posteriors = em_step(hmm)
        log_probs.append(batch_posteriors.marginal_loglik.sum())

    return hmm, log_probs


def _loss_fn(hmm, params, batch_emissions, lens):
    """Default objective function."""
    hmm.unconstrained_params = params
    f = lambda emissions, t: -hmm.marginal_log_prob(emissions) / t
    return vmap(f)(batch_emissions, lens).mean() + hmm.prior_log_prob()


def _sample_minibatches(key, sequences, lens, batch_size, shuffle):
    """Sequence generator."""
    n_seq = len(sequences)
    perm = jnp.where(shuffle, jr.permutation(key, n_seq), jnp.arange(n_seq))
    _sequences = sequences[perm]
    _lens = lens[perm]

    for idx in range(0, n_seq, batch_size):
        yield _sequences[idx:min(idx + batch_size, n_seq)], _lens[idx:min(idx + batch_size, n_seq)]


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
        lens (chex.Array or None): num_timesteps in each independent batch emissions
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
    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)

    if lens is None:
        num_sequences, num_timesteps = batch_emissions.shape[:2]
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
    hmm.unconstrained_params = params

    return hmm, losses

def _init_suff_stats(hmm, batch_shape=()):
    """Initialize expected sufficient statistics dataclass associated with hmm.
    Each field set 0-arrays with shape (*batch_shape, hmm.num_states, ...).""" 
    return tree_map(
        lambda shp: jnp.zeros(batch_shape + shp),
        hmm.suff_stats_event_shape,
        is_leaf=lambda x: isinstance(x, tuple) # Stop tree-mapping when we get to shape tuples
    )

def hmm_fit_stochastic_em(
    hmm,
    batch_emissions,
    batch_size=1,
    num_epochs=50,
    learning_rate_asymp_frac=0.9,
    key=jr.PRNGKey(0),
):
    """
    Note that batch_emissions is initially of shape (N,T,D) where N is the
    number of independent sequences and T is the length of a sequence.
    Then, a random subset of the entire sequence with shape (B,T,D), is sampled
    at each step where B is batch size.

    TODO This only works for the models which explicitly return expected
    sufficient statistics. Does not work in the general case when an
    HMMPosterior object is returned.

    Args:
        hmm (BaseHMM): HMM class whose parameters will be estimated.
        batch_emissions (chex.Array): Independent sequences, shape (N,T,D).
        batch_size (int): Number of sequences used at each update step, B.
        num_epochs (int): number of iterations to run on shuffled minibatches
        learning_rate_asymp_frac (float): Fraction of _total_ training iterations
            (i.e. num_epochs * num_batches) at which learning rate levels off,
            under an exponential decay model. Must be in range (0,1].
        key (chex.PRNGKey): PRNG key.

    Returns:
        hmm: HMM with optimized parameters.
        log_probs: sum of marginal_loglikelihood
    """
    
    num_complete_batches, leftover = jnp.divmod(len(batch_emissions), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

    num_sequences, num_timesteps = batch_emissions.shape[:2]
    lens = jnp.ones((num_sequences,)) * num_timesteps
    scale = num_sequences / batch_size

    @jit
    def _epoch_step(carry, input):
        def _minibatch_step(carry, input):
            (hmm, rolling_stats), rate = carry, input

            emissions, _ = next(sample_generator)
            
            minibatch_stats = hmm.e_step(emissions)
            these_stats = tree_map(
                partial(jnp.sum, axis=0, keepdims=True), minibatch_stats)
            
            rolling_stats = lax.cond(
                jnp.all(rolling_stats.initial_probs==0),
                partial(tree_map, lambda _, s1:  rate * scale * s1),
                partial(tree_map, lambda s0, s1: (1-rate) * s0 + rate * scale * s1,),
                rolling_stats, these_stats
            )
            
            hmm.m_step(emissions, rolling_stats)
            return (hmm, rolling_stats), these_stats.marginal_loglik.sum()

        # ------------------------------------------------------------------
        (hmm, rolling_stats), (key, learn_rates) = carry, input

        sample_generator = \
            _sample_minibatches(key, batch_emissions, lens, batch_size, True)

        (hmm, rolling_stats), minibath_log_probs = lax.scan(
                        _minibatch_step, (hmm, rolling_stats), learn_rates)

        return (hmm, rolling_stats), minibath_log_probs[-1]
    
    # ========================================================================
    # Learning rate schedule
    schedule = optax.exponential_decay(
        init_value=1.,
        transition_steps=num_epochs*num_batches,
        decay_rate=(num_epochs*num_batches)**(-1./learning_rate_asymp_frac),
        end_value=0.
        )
    learn_rates = schedule(jnp.arange(num_epochs*num_batches))
    learn_rates = learn_rates.reshape(num_epochs, num_batches)

    # Initialize suff stats fields to 0-arrays with shape (1, hmm.num_states, ...)
    init_stats = _init_suff_stats(hmm, (1,))

    (hmm, _), log_probs = lax.scan( \
        _epoch_step, (hmm, init_stats), (jr.split(key, num_epochs), learn_rates))

    return hmm, log_probs