
#pytest test_inference_linreg.py  -rP

import chex
from typing import Callable, Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import chex
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.rebayes.utils import get_mlp_flattened_params
from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, EKFIntegrals
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.rebayes.ekf import RebayesEKF

def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)

def setup():
    n_obs = 21
    obs_var = 0.1
    x = jnp.linspace(0, 20, n_obs)
    X = x[:, None] # reshape to (T,1)
    X1 = jnp.column_stack((jnp.ones_like(x), x))  # Include column of 1s
    y = jnp.array(
        [2.486, -0.303, -4.053, -4.336, -6.174, -5.604, -3.507, -2.326, -4.638, -0.233, -1.986, 1.028, -2.264,
        -0.451, 1.167, 6.652, 4.145, 5.268, 6.34, 9.626, 14.784])
    Y = y[:, None] # reshape to (T,1)

    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 1

    # Batch Bayes solution
    posterior_prec = jnp.linalg.inv(Sigma0) + X1.T @ X1 / obs_var
    cov_batch = jnp.linalg.inv(posterior_prec)
    b = jnp.linalg.inv(Sigma0) @ mu0 + X1.T @ y / obs_var
    mu_batch = jnp.linalg.solve(posterior_prec, b)

    return X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch

def test_setup():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    print('mu batch', mu_batch)
    print('cov batch', cov_batch)

def test_kalman():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    nfeatures = X1.shape[1]
    # we use H=X1 since state is the bias and then regression weights
    lgssm = LinearGaussianSSM(state_dim = nfeatures, emission_dim = 1, input_dim = 0)
    F = jnp.eye(nfeatures) # dynamics = I
    Q = jnp.zeros((nfeatures, nfeatures))  # No parameter drift.
    R = jnp.ones((1, 1)) * obs_var

    params, _ = lgssm.initialize(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_weights=F,
        dynamics_covariance=Q,
        emission_weights=X1[:, None, :], # (t, 1, D) where D = num input features
        emission_covariance=R,
        )
    lgssm_posterior = lgssm.filter(params, Y) 
    mu_kf = lgssm_posterior.filtered_means[-1]
    cov_kf = lgssm_posterior.filtered_covariances[-1]
    print('mu kf', mu_kf)
    print('cov kf', cov_kf)
    assert allclose(mu_batch, mu_kf)
    assert allclose(cov_batch, cov_kf)

def setup_ssm():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    nfeatures = X.shape[1]
    # we pass in X not X1 since DNN has a bias term as the final weight
    
    # Define Linear Regression as MLP with no hidden layers
    input_dim, hidden_dims, output_dim = nfeatures, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)
    nparams = len(flat_params)

    fcekf_params = ParamsGGSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_function=lambda w, _: w,
        dynamics_covariance = jnp.zeros((nparams, nparams)),
        emission_mean_function = lambda w, x: apply_fn(w, x),
        emission_cov_function = lambda w, x: obs_var
    )
    return fcekf_params

def test_ekf():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    fcekf_params = setup_ssm()
    fcekf_post = conditional_moments_gaussian_filter(fcekf_params, EKFIntegrals(), Y, inputs=X)
    mu_ekf = fcekf_post.filtered_means[-1]
    cov_ekf = fcekf_post.filtered_covariances[-1]
    print('mu ekf', mu_ekf)
    print('cov ekf', cov_ekf)
    assert allclose(mu_batch, mu_ekf)
    assert allclose(cov_batch, cov_ekf)

def test_rebayes_loop():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    fcekf_params = setup_ssm()
    estimator = RebayesEKF(fcekf_params, method = 'fcekf')

    bel = estimator.initialize()
    T = X.shape[0]
    for t in range(T):
        bel = estimator.update(bel, X[t], Y[t])

    print('mu rebayes loop', bel.mean)
    print('cov rebayes loop', bel.cov)
    assert allclose(mu_batch, bel.mean)
    assert allclose(cov_batch, bel.cov)

def test_rebayes_scan():
    X, X1, Y, obs_var, mu0, Sigma0, mu_batch, cov_batch = setup()
    fcekf_params = setup_ssm()
    estimator = RebayesEKF(fcekf_params, method = 'fcekf')
   
    def callback(bel, t, x, y):
        return bel.mean

    bel, outputs = estimator.scan(X, Y, callback)
    assert outputs.shape[0] == X.shape[0]
    assert outputs.shape[1] == len(mu0)
    print('mu rebayes scan', bel.mean)
    print('cov rebayes scan', bel.cov)
    assert allclose(mu_batch, bel.mean)
    assert allclose(cov_batch, bel.cov)
