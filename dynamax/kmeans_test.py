import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.sparse as sp
import tensorflow_probability.substrates.jax.distributions as tfd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms

from dynamax.kmeans import kmeans
from dynamax.kmeans import kmeans_plusplus_initialization


def _kmeans_plusplus(key, X, n_clusters, x_squared_norms, n_local_trials=None):
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py#L161
    key0, key1 = jr.split(key)
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    initial_center = np.array(jr.choice(key0, X))

    if sp.issparse(X):
        centers[0] = initial_center.toarray()
    else:
        centers[0] = initial_center

    closest_dist_sq = euclidean_distances(centers[0, np.newaxis],
                                           X,
                                           Y_norm_squared=x_squared_norms,
                                           squared=True)

    for c in range(1, n_clusters):
        key0, key1 = jr.split(key1)
        candidate_ids = tfd.Categorical(logits=jnp.log(jnp.squeeze(closest_dist_sq))).sample(
            seed=key0, sample_shape=(n_local_trials,))
        distance_to_candidates = euclidean_distances(X[candidate_ids, :],
                                                      X,
                                                      Y_norm_squared=x_squared_norms,
                                                      squared=True)

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers


def test_kmeans(seed=0,
                num_samples=1000,
                dim=100,
                variance=10.,
                num_clusters=2,
                max_iter=50,
                threshold=1e-5):

    key = jr.PRNGKey(seed)
    key0, key1 = jr.split(key)

    X = jr.normal(key0, shape=(num_samples, dim)) * variance
    initial_centroids = jr.choice(key1, X, shape=(num_clusters,))
    sklearn_centroids = KMeans(n_clusters=num_clusters,
                               random_state=0,
                               max_iter=max_iter,
                               tol=threshold,
                               init=np.array(initial_centroids),
                               n_init=1).fit(np.array(X)).cluster_centers_

    centroids, _ = kmeans(X,
                          num_clusters,
                          max_iter=max_iter,
                          threshold=threshold,
                          initial_centroids=initial_centroids)
    assert jnp.allclose(jnp.sort(centroids, axis=0), jnp.sort(sklearn_centroids, axis=0), atol=1e-6)


def test_kmeans_plus_plus(seed=0,
                          num_samples=1000,
                          dim=100,
                          variance=10.,
                          num_clusters=5,
                          max_iter=50,
                          threshold=1e-5):
    X = jr.normal(jr.PRNGKey(0), shape=(num_samples, dim)) * variance
    x_squared_norms = row_norms(np.array(X), squared=True)
    sklearn_centers = _kmeans_plusplus(jr.PRNGKey(0),
                                       np.array(X),
                                       num_clusters,
                                       x_squared_norms,
                                       n_local_trials=None)
    centers = kmeans_plusplus_initialization(jr.PRNGKey(0), X, num_clusters, num_local_trials=None)

    assert jnp.allclose(sklearn_centers, centers)
