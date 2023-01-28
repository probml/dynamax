
#pytest test_ekf2.py  -rP
# Test inference for Bayesian linear regression with static parameters

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

from dynamax.linear_gaussian_ssm import LinearGaussianSSM

from dynamax.rebayes.utils import get_mlp_flattened_params
from dynamax.rebayes.ekf2 import RebayesEKF
from dynamax.rebayes.base import *

def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)

def make_data():
    n_obs = 21
    x = jnp.linspace(0, 20, n_obs)
    X = x[:, None] # reshape to (T,1)
    X1 = jnp.column_stack((jnp.ones_like(x), x))  # Include column of 1s
    y = jnp.array(
        [2.486, -0.303, -4.053, -4.336, -6.174, -5.604, -3.507, -2.326, -4.638, -0.233, -1.986, 1.028, -2.264,
        -0.451, 1.167, 6.652, 4.145, 5.268, 6.34, 9.626, 14.784])
    Y = y[:, None] # reshape to (T,1)
    return X, Y

def make_params():
    obs_var = 0.1
    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 1
    return (obs_var, mu0, Sigma0)

def batch_bayes(X,Y):
    N = X.shape[0]
    X1 = jnp.column_stack((jnp.ones(N), X))  # Include column of 1s
    y = Y[:,0] # extract column vector
    (obs_var, mu0, Sigma0) = make_params()
    posterior_prec = jnp.linalg.inv(Sigma0) + X1.T @ X1 / obs_var
    cov_batch = jnp.linalg.inv(posterior_prec)
    b = jnp.linalg.inv(Sigma0) @ mu0 + X1.T @ y / obs_var
    mu_batch = jnp.linalg.solve(posterior_prec, b)
    return mu_batch, cov_batch


def run_kalman():
    X, Y = make_data()
    N = X.shape[0]
    X1 = jnp.column_stack((jnp.ones(N), X))  # Include column of 1s
    (obs_var, mu0, Sigma0) = make_params()
    nfeatures = X1.shape[1]
    # we use H=X1 since z=(b, w), so z'u = (b w)' (1 x)
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
    return lgssm_posterior


def test_kalman():
    X, Y = make_data()
    lgssm_posterior = run_kalman()
    mu_kf = lgssm_posterior.filtered_means[-1]
    cov_kf = lgssm_posterior.filtered_covariances[-1]
    mu_batch, cov_batch = batch_bayes(X,Y)
    assert allclose(mu_batch, mu_kf)
    assert allclose(cov_batch, cov_kf)

def setup_ssm():
    X, Y = make_data()
    (obs_var, mu0, Sigma0) = make_params()
    nfeatures = X.shape[1]
    # we pass in X not X1 since DNN has a bias term 
    
    # Define Linear Regression as MLP with no hidden layers
    input_dim, hidden_dims, output_dim = nfeatures, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)
    nparams = len(flat_params)
    
    params = RebayesParams(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_weights = jnp.eye(nparams),
        dynamics_covariance = jnp.zeros((nparams, nparams)),
        emission_mean_function = lambda w, x: apply_fn(w, x),
        emission_cov_function = lambda w, x: obs_var
    )

    return params


def test_rebayes_loop():
    (X, Y) = make_data()
    params  = setup_ssm()
    estimator = RebayesEKF(params, method = 'fcekf')

    lgssm_posterior = run_kalman()
    mu_kf = lgssm_posterior.filtered_means
    cov_kf = lgssm_posterior.filtered_covariances
    ll_kf = lgssm_posterior.marginal_loglik
    def callback(pred_obs, bel, t, u, y):
        m, P = pred_obs.mean, pred_obs.cov
        ll = MVN(m, P).log_prob(jnp.atleast_1d(y))
        assert allclose(bel.mean, mu_kf[t])
        assert allclose(bel.cov, cov_kf[t])
        return ll

    bel = estimator.init_bel()
    T = X.shape[0]
    ll = 0
    for t in range(T):
        pred_obs = estimator.predict_obs(bel, X[t])
        bel = estimator.predict_state(bel, X[t])
        bel = estimator.update_state(bel, X[t], Y[t]) 
        ll += callback(pred_obs, bel, t, X[t], Y[t])  
    assert jnp.allclose(ll, ll_kf, atol=1e-1)


def broken_test_rebayes_scan():
    (X, Y) = make_data()
    params  = setup_ssm()
    estimator = RebayesEKF(params, method = 'fcekf')

    lgssm_posterior = run_kalman()
    mu_kf = lgssm_posterior.filtered_means
    cov_kf = lgssm_posterior.filtered_covariances
    ll_kf = lgssm_posterior.marginal_loglik

    def callback(pred_obs, bel, t, u, y):
        m, P = pred_obs.mean, pred_obs.cov
        ll = MVN(m, P).log_prob(jnp.atleast_1d(y))
        return ll

    final_bel, lls = estimator.scan(X, Y,  callback)
    T = mu_kf.shape[0]
    assert allclose(final_bel.mean, mu_kf[T-1])
    assert allclose(final_bel.cov, cov_kf[T-1])
    print(lls)
    ll = jnp.sum(lls)
    assert jnp.allclose(ll, ll_kf, atol=1e-1)