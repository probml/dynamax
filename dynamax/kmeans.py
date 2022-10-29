from functools import partial

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import jit
from jax import lax
from jax import vmap


def find_closest_cluster(X, centroids):

    def assign(x):
        distances = vmap(jnp.linalg.norm)(centroids - x)
        label = jnp.argmin(distances)
        min_distance = distances[label]
        return label, min_distance

    return vmap(assign)(X)


def kmeans_plusplus_initialization(key, X, num_clusters, num_local_trials=None):
    """Computational component for initialization of num_clusters by
    k-means++. Prior validation of data is assumed.
    https://github.com/scikit-learn/scikit-learn/blob/138619ae0b421826cf838af24627150fa8684cf5/sklearn/cluster/_kmeans.py#L161
    Parameters
    ----------
    x : (num_samples, num_features)
        The data to pick seeds for.
    num_clusters : int
        The number of seeds to choose.
    num_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (num_clusters, num_features)
        The initial centers for k-means.
    """
    num_samples, num_features = X.shape
    num_clusters = min(num_clusters, num_samples)
    key0, key1 = jr.split(key)

    # Set the number of local seeding trials if none is given
    if num_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        num_local_trials = 2 + jnp.log(num_clusters).astype(jnp.int32)

    # Pick first center randomly and track index of point
    initial_center = jr.choice(key0, X)[None, ...]

    def euclidean_distance_square(x, y):
        return jnp.square(x - y).sum(axis=-1)

    # Initialize list of closest distances and calculate current potential
    initial_distances = euclidean_distance_square(X, initial_center)

    # Pick the remaining n_clusters-1 points
    def find_center(carry, i):
        distances, key = carry
        key0, key1 = jr.split(key)
        candidate_ids = tfd.Categorical(logits=jnp.log(distances)).sample(
            seed=key0, sample_shape=(num_local_trials,))

        # XXX: numerical imprecision can result in a candidate_id out of range
        candidate_ids = jnp.clip(candidate_ids, a_max=num_samples - 1)

        # Compute distances to center candidates
        distance_to_candidates = vmap(lambda x, y: euclidean_distance_square(x[None, ...], y),
                                      in_axes=(0, None))(X[candidate_ids], X)
        # update closest distances squared and potential for each candidate
        distance_to_candidates = vmap(jnp.minimum, in_axes=(0, None))(distance_to_candidates,
                                                                      distances)

        # Decide which candidate is the best
        best_candidate = jnp.argmin(jnp.sum(distance_to_candidates, axis=-1))
        distances = distance_to_candidates[best_candidate]
        candidate_id = candidate_ids[best_candidate]

        return (distances, key1), X[candidate_id]

    _, centers = lax.scan(find_center, (initial_distances, key1), None, length=num_clusters - 1)
    centers = jnp.vstack([initial_center, centers])
    return centers


@partial(jit, static_argnums=(1,))
def kmeans(X,
           num_clusters,
           max_iter=50,
           threshold=1e-4,
           initial_centroids=None,
           num_init_iterations=1,
           key=jr.PRNGKey(0)):
    # https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA#scrollTo=XUaIhb7TmtGo
    num_clusters = min(len(X), num_clusters)

    def improve_centroids(state):
        prev_centroids, prev_dists, _, i = state
        assignment, distortions = find_closest_cluster(X, prev_centroids)

        # Clip to change 0/0 later to 0/1
        counts = jnp.clip(
            (assignment[None, :] == jnp.arange(num_clusters)[:, None]).sum(axis=1, keepdims=True),
            a_min=1.)

        # Sum over points in a centroid by zeroing others out
        new_centroids = jnp.sum(
            jnp.where(
                # axes: (data points, clusters, data dimension)
                assignment[:, None, None] == jnp.arange(num_clusters)[None, :, None],
                X[:, None, :],
                0.,
            ),
            axis=0,
        ) / counts

        return new_centroids, jnp.mean(distortions), prev_dists, i + 1

    # Run one iteration to initialize distortions
    if initial_centroids is None:
        initial_centroids = jr.choice(key, X, shape=(num_clusters,))

    def init_centroids(state, i):
        state = improve_centroids(state)
        return state, None

    initial_state, _ = lax.scan(init_centroids, (initial_centroids, jnp.inf, 0, 0),
                                jnp.arange(num_init_iterations))

    # Iterate until convergence
    centroids, distortions, *_ = lax.while_loop(
        lambda val: ((val[2] - val[1]) > threshold) & (val[3] < max_iter),
        improve_centroids,
        initial_state,
    )
    return centroids, distortions
