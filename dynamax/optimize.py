import jax.numpy as jnp
import jax.random as jr
import optax
from jax import lax, value_and_grad
from jax.tree_util import tree_map
from dynamax.utils import pytree_len


def sample_minibatches(key, dataset, batch_size, shuffle):
    """Sequence generator.

    NB: The generator does not preform as expected when used to yield data
        within jit'd code. This is likely because the generator internally
        updates a state with each yield (which doesn't play well with jit).
    """
    n_data = pytree_len(dataset)
    perm = jnp.where(shuffle, jr.permutation(key, n_data), jnp.arange(n_data))
    for idx in range(0, n_data, batch_size):
        yield tree_map(lambda x: x[perm[idx:min(idx + batch_size, n_data)]], dataset)


def run_sgd(loss_fn,
            params,
            dataset,
            optimizer=optax.adam(1e-3),
            batch_size=1,
            num_epochs=50,
            shuffle=False,
            key=jr.PRNGKey(0)):
    """
    Note that batch_emissions is initially of shape (N,T)
    where N is the number of independent sequences and
    T is the length of a sequence. Then, a random susbet with shape (B, T)
    of entire sequence, not time steps, is sampled at each step where B is
    batch size.

    Args:
        loss_fn (Callable): Objective function.
        params (PyTree): initial value of parameters to be estimated.
        dataset (chex.Array): PyTree of data arrays with leading batch dimension
        optmizer (optax.Optimizer): Optimizer.
        batch_size (int): Number of sequences used at each update step.
        num_iters (int): Iterations made on only one mini-batch.
        shuffle (bool): Indicates whether to shuffle emissions.
        key (chex.PRNGKey): RNG key.

    Returns:
        hmm: HMM with optimized parameters.
        losses: Output of loss_fn stored at each step.
    """
    opt_state = optimizer.init(params)
    num_batches = pytree_len(dataset)
    num_complete_batches, leftover = jnp.divmod(num_batches, batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
    loss_grad_fn = value_and_grad(loss_fn)

    if batch_size >= num_batches:
        shuffle = False

    def train_step(carry, key):
        params, opt_state = carry
        sample_generator = sample_minibatches(key, dataset, batch_size, shuffle)

        def cond_fun(state):
            itr, params, opt_state, avg_loss = state
            return itr < num_batches

        def body_fun(state):
            itr, params, opt_state, avg_loss = state
            minibatch = next(sample_generator)  ## TODO: Does this work inside while_loop??
            this_loss, grads = loss_grad_fn(params, minibatch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return itr + 1, params, opt_state, (avg_loss * itr + this_loss) / (itr + 1)

        init_val = (0, params, opt_state, 0.0)
        _, params, opt_state, avg_loss = lax.while_loop(cond_fun, body_fun, init_val)
        return (params, opt_state), avg_loss

    keys = jr.split(key, num_epochs)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)
    return params, losses


def run_gradient_descent(loss_fn,
                         params,
                         batch_emissions,
                         optimizer=optax.adam(1e-3),
                         num_epochs=50,
                         key=jr.PRNGKey(0),
                         **batch_covariates):
    """
    Note that batch_emissions is initially of shape (N,T)
    where N is the number of independent sequences in the batch and
    T is the length of each sequence.

    Args:
        loss_fn (Callable): Objective function.
        params (PyTree): initial value of parameters to be estimated.
        dataset (chex.Array): PyTree of data arrays with leading batch dimension
        optmizer (optax.Optimizer): Optimizer.
        num_iters (int): Iterations made on only one mini-batch.
        key (chex.PRNGKey): RNG key.

    Returns:
        hmm: HMM with optimized parameters.
        losses: Output of loss_fn stored at each step.
    """
    opt_state = optimizer.init(params)
    loss_grad_fn = value_and_grad(loss_fn)

    def train_step(carry):
        params, opt_state = carry
        loss, grads = loss_grad_fn(params, batch_emissions, **batch_covariates)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    keys = jr.split(key, num_epochs)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)
    return params, losses
