import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
from jax import vmap, jit, lax, value_and_grad
import jax.random as jr
from jax.tree_util import tree_map
import optax

# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
PSDToRealBijector = tfb.Chain(
    [
        tfb.Invert(tfb.FillTriangular()),
        tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
        tfb.Invert(tfb.CholeskyOuterProduct()),
    ]
)


@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    """
    Pad ragged sequences to a fixed length.
    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences
    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence
    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced
    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    """

    def pad(seq, len):
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)

    dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return dataset


def _sample_minibatches(key,
                        batch_data,
                        batch_args,
                        batch_kwargs,
                        batch_size,
                        shuffle):
    """Sequence generator."""
    n_seq = len(batch_data)
    perm = jnp.where(shuffle, jr.permutation(key, n_seq), jnp.arange(n_seq))
    _data = tree_map(lambda x: x[perm], batch_data)
    _args = tree_map(lambda x: x[perm], batch_args)
    _kwargs = tree_map(lambda x: x[perm], batch_kwargs)

    for idx in range(0, n_seq, batch_size):
        yield (tree_map(lambda x: x[idx:min(idx + batch_size, n_seq)], _data),
               tree_map(lambda x: x[idx:min(idx + batch_size, n_seq)], _args),
               tree_map(lambda x: x[idx:min(idx + batch_size, n_seq)], _kwargs))


def sgd_helper(
    loss_fn,
    params,
    batch_data,
    batch_args=(),
    batch_kwargs={},
    optimizer=optax.adam(1e-3),
    batch_size=1,
    num_iters=50,
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
        params: unconstrained parameters to be optimized
        batch_emissions (chex.Array): Independent sequences.
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

    if batch_size == len(batch_data):
        shuffle = False

    num_complete_batches, leftover = jnp.divmod(len(batch_data), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

    loss_grad_fn = value_and_grad(loss_fn)
    def train_step(carry, key):

        sample_generator = _sample_minibatches(
            key, batch_data, batch_args, batch_kwargs, batch_size, shuffle)

        def cond_fun(args):
            itr, _, _, _ = args
            return itr < num_batches

        def body_fun(args):
            itr, loss, params, opt_state = args
            minibatch_data, minibatch_args, minibatch_kwargs = next(sample_generator)
            this_loss, grads = loss_grad_fn(params,
                                            minibatch_data,
                                            *minibatch_args,
                                            **minibatch_kwargs)

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return itr + 1, loss + this_loss, params, opt_state

        params, opt_state = carry
        (_, loss, params, opt_state) = lax.while_loop(cond_fun, body_fun, (0, 0.0, params, opt_state))
        return (params, opt_state), loss

    keys = jr.split(key, num_iters)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)

        # sample_generator = _sample_minibatches(
        #     key, batch_emissions, batch_covariates, batch_size, shuffle)

        # def opt_step(carry, i):
        #     params, opt_state = carry
        #     batch, covariates = next(sample_generator)
        #     val, grads = loss_grad_fn(params, batch, **covariates)
        #     updates, opt_state = optimizer.update(grads, opt_state)
        #     params = optax.apply_updates(params, updates)
        #     return (params, opt_state), val

        # state, losses = lax.scan(opt_step, carry, jnp.arange(num_batches))
        # return state, losses.mean()

    # keys = jr.split(key, num_iters)
    # (params, _), losses = lax.scan(train_step, (params, opt_state), keys)
    # losses = losses.flatten()

    return params, losses
