import jax
from jax import tree_leaves, vmap, lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.random as jr
from jax.tree_util import tree_map, tree_leaves

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from chex import Array
import chex


def ess_criterion(log_weights: Array) -> Array:
  """A criterion that resamples based on effective sample size."""
  num_particles = log_weights.shape[0]
  ess_num = 2 * logsumexp(log_weights)
  ess_denom = logsumexp(2 * log_weights)
  log_ess = ess_num - ess_denom
  return log_ess <= jnp.log(num_particles / 2.0)


def never_resample_criterion(log_weights: Array) -> Array:
  """A criterion that never resamples."""
  del log_weights
  return jnp.array(False)


def always_resample_criterion(log_weights: Array) -> Array:
  """A criterion that always resamples."""
  del log_weights
  return jnp.array(True)


def multinomial_resampling(
    key: jr.PRNGKey, log_weights: Array, particles):
  """Resample particles with multinomial resampling.

  Args:
    key: A JAX PRNG key.
    log_weights: A [num_particles] ndarray, the log weights for each particle.
    particles: A pytree of [num_particles, ...] ndarrays that
      will be resampled.
  Returns:
    resampled_particles: A pytree of [num_particles, ...] ndarrays resampled via
      multinomial sampling.
    parents: A [num_particles] array containing index of parent of each state
  """
  num_particles = log_weights.shape[0]
  cat = tfd.Categorical(logits=log_weights)
  ancestors = cat.sample(sample_shape=(num_particles,), seed=key)
  assert isinstance(ancestors, jnp.ndarray)
  return (tree_map(lambda item: item[ancestors], particles), ancestors)


@chex.dataclass
class SMCPosterior:
    marginal_loglik: chex.Scalar
    particles: chex.Array
    log_weights: chex.Array
    ancestors: chex.Array
    resampled: chex.Array


def generic_smc(key,
                initial_particles,
                propose_and_weight,
                emissions,
                initial_log_weights=None,
                resampling_criterion=ess_criterion,
                resampling_fn=multinomial_resampling,
                **covariates):

    num_particles = len(tree_leaves(initial_particles)[0])
    if initial_log_weights is None:
        initial_log_weights = jnp.zeros(num_particles)

    def smc_step(carry, args):
        key, particles, log_weights = carry
        key, subkey1, subkey2 = jr.split(key, num=3)
        emission, covariate = args

        # Propagate the particle states
        wrapped_propose_and_weight = \
            lambda key, particle, emission, covariate_dict: \
                propose_and_weight(key, particle, emission, **covariate_dict)

        f = vmap(wrapped_propose_and_weight, in_axes=(0, 0, None, None))
        new_particles, incremental_log_weight = f(
            jr.split(subkey2, num=num_particles),
            particles,
            emission,
            covariate)

        # Update the log weights.
        log_weights += incremental_log_weight

        # Resample the particles if resampling_criterion returns True and we haven't
        # exceeded the supplied number of steps.
        should_resample = resampling_criterion(log_weights)
        resampled_particles, ancestors = lax.cond(
            should_resample,
            lambda args: resampling_fn(*args),
            lambda args: (args[2], jnp.arange(num_particles)),
            (subkey1, log_weights, new_particles)
        )

        resampled_log_weights = (1. - should_resample) * log_weights
        # NOTE - grumble grumble grumble.  0.0 * inf in the line above seems to
        #  trigger nans that break everything.
        resampled_log_weights = lax.cond(
            should_resample,
            lambda _ws: jnp.nan_to_num(_ws),
            lambda _ws: _ws,
            resampled_log_weights)

        return ((key, resampled_particles, resampled_log_weights),
                (new_particles, log_weights, ancestors, should_resample))

    # Scan over emissions and covariates
    _, (particles, log_weights, ancestors, resampled) = jax.lax.scan(
        smc_step,
        (key, initial_particles, initial_log_weights),
        (emissions, covariates))

    # Average along particle dimension
    log_avg_weights = logsumexp(log_weights, axis=1) - jnp.log(num_particles)

    # Sum in time dimension on resampling steps.
    # Note that this does not include any steps past num_steps because
    # the resampling criterion doesn't allow resampling past num_steps steps.
    # If we didn't resample on the last timestep, add in the missing log_p_hat
    marginal_loglik = jnp.sum(log_avg_weights * resampled)
    marginal_loglik += jnp.where(resampled[-1], 0., log_avg_weights[-1])

    return SMCPosterior(
        marginal_loglik=marginal_loglik,
        particles=particles,
        log_weights=log_weights,
        ancestors=ancestors,
        resampled=resampled)


def smc(key,
        initial_distribution,
        transition_distribution,
        emission_distribution,
        proposal_distribution,
        emissions,
        num_particles=1,
        resampling_criterion=ess_criterion,
        resampling_fn=multinomial_resampling,
        **covariates):


    def propose_and_weight(key, particle, emission, **covariate):
        # sample the proposal distribution
        q = proposal_distribution(particle, emission, **covariate)
        next_particle = q.sample(seed=key)

        # evaluate the incremental log weight
        log_weight = transition_distribution(particle, **covariate).log_prob(next_particle)
        log_weight += emission_distribution(next_particle, **covariate).log_prob(emission)
        log_weight -= q.log_prob(next_particle)
        return next_particle, log_weight

    # Sample initial particles from initial distribution
    # TODO: We could have a proposal for the initial particles too
    initial_key, key = jr.split(key, 2)
    initial_covariate = tree_map(lambda x: x[0], covariates)
    initial_emission = tree_map(lambda x: x[0], emissions)
    initial_particles = initial_distribution(**initial_covariate).sample(seed=initial_key, sample_shape=(num_particles,))
    initial_log_weights = vmap(lambda x: emission_distribution(x, **initial_covariate).log_prob(initial_emission))(initial_particles)

    return generic_smc(key,
                       initial_particles,
                       propose_and_weight,
                       emissions,
                       initial_log_weights=initial_log_weights,
                       resampling_criterion=resampling_criterion,
                       resampling_fn=resampling_fn,
                       **covariates)


def reconstruct_particle_trajectories(posterior):
    """Computes the resampled SMC states.

    Args:
    states: A PyTree with leaves of leading dimensions [max_num_timesteps, num_particles].
    ancestor_inds: The ancestor indices returned by smc, an Array of shape
        [max_num_timesteps, num_particles].
    resampled: An Array of shape [max_num_timesteps] indicating if smc resampled on each timestep.
    num_timesteps: The number of timesteps smc ran for, can be less than max_num_timesteps.
    Returns:
    A Pytree with the same structure and leaf node shape of states containing the resampled
        states.
    """
    _, num_particles = posterior.ancestors.shape

    def scan_fn(inds, ancestors):
        new_inds = ancestors[inds]
        return new_inds, new_inds

    init_inds = jnp.arange(num_particles)
    _, inds = jax.lax.scan(
        scan_fn,
        init_inds,
        posterior.ancestors[:-1],     # TODO: Check this indexing
        reverse=True)
    inds = jnp.concatenate([inds, init_inds[jnp.newaxis]])

    def map_fn(x):
        extra_state_dims = x.ndim - 2
        new_inds_shape = [inds.shape[0], inds.shape[1]] + [1] * extra_state_dims
        new_inds = jnp.reshape(inds, new_inds_shape)
        return jnp.take_along_axis(x, new_inds, axis=1)

    return tree_map(map_fn, posterior.particles)
