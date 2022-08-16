import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow as tf
from jax import jit
from jax import value_and_grad
from jax.tree_util import tree_leaves


def _get_dataset_len(dataset):
    return len(tree_leaves(dataset)[0])


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
        dataset (chex.Array or tf.data.Dataset): PyTree of data arrays with leading batch dimension
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
    loss_grad_fn = value_and_grad(loss_fn)

    dataset_len = _get_dataset_len(dataset)

    if isinstance(dataset, jnp.DeviceArray):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if batch_size >= dataset_len:
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

        if shuffle:
            dataset = dataset.shuffle(dataset_len)

        for minibatch in dataset.batch(batch_size):
            params, opt_state, loss = update(params, opt_state, jnp.asarray(minibatch))
            losses_per_batch.append(loss)

        losses.append(losses_per_batch)

    losses = jnp.array(losses).mean(axis=-1)
    return params, losses
