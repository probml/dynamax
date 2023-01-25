import pytest
import numpy as np
import jax.numpy as jnp
import time

from dynamax.utils import datasets
from dynamax.rebayes.utils import get_mlp_flattened_params
from dynamax.rebayes.orfit import ORFitParams, orthogonal_recursive_fitting, RebayesORFit
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import EKFIntegrals, conditional_moments_gaussian_filter


def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)


def load_rmnist_data(num_train=100):
    # Load rotated MNIST dataset
    np.random.seed(314)

    train, _ = datasets.load_rotated_mnist(target_digit=2)
    X_train, y_train = train

    X_train = jnp.array(X_train)[:num_train]
    y_train = jnp.array(y_train)[:num_train]

    X_train = (X_train - X_train.mean()) / X_train.std()

    return X_train, y_train


def setup_orfit(memory_size):
    # Define Linear Regression as single layer perceptron
    input_dim, hidden_dims, output_dim = 784, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

    # L2 loss
    loss_fn = lambda w, x, y: (apply_fn(w, x) - y)**2

    # Define ORFit parameters
    orfit_params = ORFitParams(
        initial_mean=flat_params,
        apply_function=apply_fn,
        loss_function=loss_fn,
        memory_size=memory_size,
    )
    return orfit_params


def setup_kf():
    # Define Linear Regression as single layer perceptron
    input_dim, hidden_dims, output_dim = 784, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

    # Define Kalman Filter parameters
    kf_params = ParamsGGSSM(
        initial_mean=flat_params,
        initial_covariance=jnp.eye(len(flat_params)),
        dynamics_function=lambda w, x: w,
        dynamics_covariance=jnp.zeros((len(flat_params), len(flat_params))),
        emission_mean_function=apply_fn,
        emission_cov_function=lambda w, x: jnp.array([0.]),
    )
    return kf_params


def test_orfit():
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = load_rmnist_data(n_train)

    # Run Infinite-memory ORFit
    orfit_params = setup_orfit(n_train)
    orfit_before_time = time.time()
    orfit_posterior = orthogonal_recursive_fitting(orfit_params, y_train, X_train)
    orfit_after_time = time.time()
    # print(f"ORFit took {orfit_after_time - orfit_before_time} seconds.")

    # Run Kalman Filter
    kf_params = setup_kf()
    kf_before_time = time.time()
    kf_posterior = conditional_moments_gaussian_filter(kf_params, EKFIntegrals(), y_train, inputs=X_train)
    kf_after_time = time.time()
    # print(f"Kalman Filter took {kf_after_time - kf_before_time} seconds.")

    assert allclose(orfit_posterior.filtered_means, kf_posterior.filtered_means)


def test_rebayes_orfit_loop():
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = load_rmnist_data(n_train)

    # Run Infinite-memory ORFit
    orfit_params = setup_orfit(n_train)
    estimator = RebayesORFit(orfit_params)
    orfit_before_time = time.time()
    bel = estimator.initialize()
    for i in range(n_train):
        bel = estimator.update(bel, X_train[i], y_train[i])
    orfit_after_time = time.time()
    print(f"Looped ORFit took {orfit_after_time - orfit_before_time} seconds.")

    # Run Kalman Filter
    kf_params = setup_kf()
    kf_before_time = time.time()
    kf_posterior = conditional_moments_gaussian_filter(kf_params, EKFIntegrals(), y_train, inputs=X_train)
    kf_after_time = time.time()
    # print(f"Kalman Filter took {kf_after_time - kf_before_time} seconds.")

    assert allclose(bel.mean, kf_posterior.filtered_means[-1])


def test_rebayes_orfit_scan():
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = load_rmnist_data(n_train)

    # Run Infinite-memory ORFit
    orfit_params = setup_orfit(n_train)
    estimator = RebayesORFit(orfit_params)
    def callback(bel, t, x, y):
        return bel.mean
    orfit_before_time = time.time()
    bel, outputs = estimator.scan(X_train, y_train, callback)
    orfit_after_time = time.time()
    print(f"Scanned ORFit took {orfit_after_time - orfit_before_time} seconds.")

    # Run Kalman Filter
    kf_params = setup_kf()
    kf_before_time = time.time()
    kf_posterior = conditional_moments_gaussian_filter(kf_params, EKFIntegrals(), y_train, inputs=X_train)
    kf_after_time = time.time()
    # print(f"Kalman Filter took {kf_after_time - kf_before_time} seconds.")

    assert allclose(bel.mean, kf_posterior.filtered_means[-1])