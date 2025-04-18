"""
This module contains functions for optimization.
"""
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import jit, lax, value_and_grad
from jax.tree_util import tree_map
from dynamax.utils.utils import pytree_len


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
        loss_fn: Objective function.
        params: initial value of parameters to be estimated.
        dataset: PyTree of data arrays with leading batch dimension
        optmizer: Optimizer.
        batch_size: Number of sequences used at each update step.
        num_iters: Iterations made on only one mini-batch.
        shuffle: Indicates whether to shuffle emissions.
        key: RNG key.

    Returns:
        params: The optimized parameters giving a low loss.
        losses: Output of loss_fn stored at each step.
    """
    opt_state = optimizer.init(params)
    num_batches = pytree_len(dataset)
    num_complete_batches, leftover = jnp.divmod(num_batches, batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
    loss_grad_fn = jit(value_and_grad(loss_fn))

    if batch_size >= num_batches:
        shuffle = False

    keys = jr.split(key, num_epochs)
    losses = []
    for key in keys:
        sample_generator = sample_minibatches(key, dataset, batch_size, shuffle)
        avg_loss = 0.0
        for itr in range(num_batches):
            minibatch = next(sample_generator)
            this_loss, grads = loss_grad_fn(params, minibatch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            avg_loss = (avg_loss * itr + this_loss) / (itr + 1)
        losses.append(avg_loss)
    return params, jnp.stack(losses)


def run_gradient_descent(objective,
                         params,
                         optimizer=optax.adam(1e-2),
                         optimizer_state=None,
                         num_mstep_iters=50):
    """
    Run gradient descent on the objective function.
    """
    if optimizer_state is None:
        optimizer_state = optimizer.init(params)

    # Minimize the negative expected log joint with gradient descent
    loss_grad_fn = value_and_grad(objective)

    # One step of the algorithm
    def train_step(carry, args):
        """One step of the algorithm."""
        params, optimizer_state = carry
        loss, grads = loss_grad_fn(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return (params, optimizer_state), loss

    # Run the optimizer
    initial_carry =  (params, optimizer_state)
    (params, optimizer_state), losses = \
        lax.scan(train_step, initial_carry, None, length=num_mstep_iters)

    # Return the updated parameters
    return params, optimizer_state, losses
