import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit
from jax import value_and_grad
from jax.tree_util import tree_leaves
from jax.tree_util import tree_map


def _get_dataset_len(dataset):
    return len(tree_leaves(dataset)[0])


def sample_minibatches(key, dataset, batch_size, shuffle):
    """Sequence generator.
    
    NB: The generator does not preform as expected when used to yield data
        within jit'd code. This is likely because the generator internally
        updates a state with each yield (which doesn't play well with jit).
    """
    n_data = _get_dataset_len(dataset)
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
            key=jr.PRNGKey(0),
            **batch_covariates):
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
    num_complete_batches, leftover = jnp.divmod(len(dataset), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
    loss_grad_fn = value_and_grad(loss_fn)

    if batch_size >= _get_dataset_len(dataset):
        shuffle = False

    @jit
    def update(params, opt_state, minibatch):
        loss, grads = loss_grad_fn(params, minibatch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    keys = jr.split(key, num_epochs)

    for key in keys:
        losses_per_batch = []
        sample_generator = sample_minibatches(key, dataset, batch_size, shuffle)

        for minibatch in sample_generator:
            params, opt_state, loss = update(params, opt_state, minibatch)
            losses_per_batch.append(loss)

        losses.append(losses_per_batch)

    losses = jnp.array(losses).mean(axis=-1)
    return params, losses
