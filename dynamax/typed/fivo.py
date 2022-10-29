from abc import abstractmethod
import equinox as eqx

from typing import Tuple, Protocol, TypeVar, Generic
import jax
import jax.numpy as jnp
from jax._src.random import KeyArray as PRNGKey


from chex import Array, ArrayTree, Scalar
import tensorflow_probability.substrates.jax as tfp

tf = tfp.tf2jax
tfd = tfp.distributions
tfd_e = tfp.experimental.distributions

LatentType = TypeVar('LatentType', bound=ArrayTree)
ObsType = TypeVar('ObsType', bound=ArrayTree)


class SSM(eqx.Module, Generic[LatentType, ObsType]):

  @abstractmethod
  def initial_state_prior(self) -> tfd.Distribution:
    """Compute the distribution over the initial latent state, z_0."""
    ...

  @abstractmethod
  def dynamics_dist(self, prev_latent: LatentType, t: int) -> tfd.Distribution:
    ...

  @abstractmethod
  def emission_dist(self, cur_latent: LatentType, t: int) -> tfd.Distribution:
    ...

  def sample_trajectory(self, key: PRNGKey, seq_len: int) -> Tuple[LatentType, ObsType]:
    """Samples a trajectory from the model.

    Args:
      key: A JAX PRNGKey.
      seq_len: The number of steps to sample.
    Returns:
      latents: The latents, an ArrayTree with leaves of shape [num_timesteps, ...].
      observations: The observations, an ArrayTree of shape [num_timesteps, ...].
    """
    key, subkey = jax.random.split(key)
    dummy_init_z = self.initial_state_prior().sample(seed=subkey)

    def scan_fn(
            carry: Tuple[PRNGKey, LatentType],
            t: int) -> Tuple[Tuple[PRNGKey, LatentType], Tuple[LatentType, ObsType]]:
      key, prev_z = carry
      key, sk1, sk2 = jax.random.split(key, num=3)
      new_z = jax.lax.cond(t == 0,
              lambda args: self.initial_state_prior().sample(seed=args[1]),
              lambda args: self.dynamics_dist(args[0], t).sample(seed=args[1]),
              (prev_z, sk1))
      new_x = self.emission_dist(new_z, t).sample(seed=sk2)
      return (key, new_z), (new_z, new_x)

    _, samples = jax.lax.scan(scan_fn, (key, dummy_init_z), jnp.arange(seq_len))

    return samples

  def log_prob(self, latents: LatentType, observations: ObsType) -> Scalar:
    """Compute the joint prob of a sequence of latents and observations, p(x_{1:T}, z_{1:T})."""
    seq_len = jax.tree_util.tree_flatten(observations)[0][0].shape[0]
    dummy_init_latent = self.initial_state_prior().sample(seed=jax.random.PRNGKey(0))

    def scan_fn(carry, cur_state):
      prev_z, log_p = carry
      cur_z, cur_x, t = cur_state
      cur_log_p_z = jax.lax.cond(t == 0,
              lambda _: self.initial_state_prior().log_prob(cur_z),
              lambda prev_z: self.dynamics_dist(prev_z, t).log_prob(cur_z),
              prev_z)
      cur_log_p_x = self.emission_dist(cur_z, t).log_prob(cur_x)
      return (cur_z, log_p + cur_log_p_z + cur_log_p_x), None

    (_, log_p), _ = jax.lax.scan(
        scan_fn,
        (dummy_init_latent, 0.),
        (latents, observations, jnp.arange(seq_len)),
        length=seq_len)

    return log_p


class SequenceModel(Protocol[LatentType, ObsType]):

  def sample_trajectory(self, __key: PRNGKey, __seq_len: int) -> Tuple[LatentType, ObsType]:
    ...

  def log_prob(self, __latents: LatentType, __observations: ObsType) -> Scalar:
    ...


class Proposal(Protocol[LatentType]):

  def propose_and_weight(self,
        __key: PRNGKey,
        __prev_latent: LatentType,
        __cur_obs: Array,
        __t: int) -> Tuple[LatentType, Scalar]:
    ...