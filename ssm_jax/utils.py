import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
from jax import vmap, jit

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

def one_hot(z, K):
    z = jnp.atleast_1d(z).astype(int)
    # assert jnp.all(z >= 0) and jnp.all(z < K)
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.arange(K)[jnp.ravel(z)]].set(1)
    zoh = jnp.reshape(zoh, shp + (K,))
    return zoh
