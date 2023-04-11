import numpy as np
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
from functools import partial
import chex
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad, jit
from jax.tree_util import tree_map, tree_reduce
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import time
import platform

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataclasses import dataclass
from itertools import cycle

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import torch
#import torch_xla.core.xla_model as xm

from dynamax.linear_gaussian_ssm import LinearGaussianSSM


def allclose(u, v):
    # we cast to numpy so we can compare pytorch and jax
    return np.allclose(np.array(u), np.array(v), atol=1e-3)

### Data

def make_linreg_data(N, D, key=jr.PRNGKey(0)):
    keys = jr.split(key, 3)
    X = jr.normal(keys[0], (N, D))
    w = jr.normal(keys[1], (D, 1))
    y = X @ w + 0.1*jr.normal(keys[2], (N, 1))
    return X, y

def make_linreg_prior(D):
    obs_var = 0.1
    mu0 = jnp.zeros(D)
    Sigma0 = jnp.eye(D) * 1
    return (obs_var, mu0, Sigma0)


def make_params_and_data(N, D, key=jr.PRNGKey(0)):
    X, Y = make_linreg_data(N, D, key)
    X1 = jnp.column_stack((jnp.ones(N), X))  # Include column of 1s
    nfeatures = X1.shape[1]
    (obs_var, mu0, Sigma0) = make_linreg_prior(nfeatures)
    F = jnp.eye(nfeatures) # dynamics = I
    Q = jnp.zeros((nfeatures, nfeatures))  # No parameter drift.
    R = jnp.ones((1, 1)) * obs_var
    Ht = X1[:, None, :] # (T,D) -> (T,1,D), H[t]'z = (b w)' (1 x)
    param_dict = {'mu0': mu0, 'Sigma0': Sigma0, 'F': F, 'Q': Q, 'R': R, 'Ht': Ht}
    return param_dict, X, Y

#### Minimal KF implementation

def predict(m, S, F, Q):
    mu_pred = F @ m 
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred

def condition_on(m, P, H, R, y):
    S = R + H @ P @ H.T
    K = jnp.linalg.solve(S + 1e-6, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - H @ m)
    return mu_cond, Sigma_cond


def kf(params, emissions, return_covs=False):
    F, Q, R = params['F'], params['Q'], params['R']
    def step(carry, t):
        ll, pred_mean, pred_cov = carry
        H = params['Ht'][t]
        y = emissions[t]
        ll += MVN(H @ pred_mean, H @ pred_cov @ H.T + R).log_prob(y)
        filtered_mean, filtered_cov = condition_on(pred_mean, pred_cov, H, R, y)
        pred_mean, pred_cov = predict(filtered_mean, filtered_cov, F, Q)
        carry = (ll, pred_mean, pred_cov)
        if return_covs:
            return carry, (filtered_mean, filtered_cov)
        else:
            return carry, (filtered_mean, None)     
    
    num_timesteps = len(emissions)
    carry = (0.0, params['mu0'], params['Sigma0'])
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(step, carry, jnp.arange(num_timesteps))
    return ll, filtered_means, filtered_covs

## Test

def batch_bayes(X,Y):
    N, D = X.shape
    X1 = jnp.column_stack((jnp.ones(N), X))  # Include column of 1s
    y = Y[:,0] # extract column vector
    (obs_var, mu0, Sigma0) = make_linreg_prior(D+1)
    posterior_prec = jnp.linalg.inv(Sigma0) + X1.T @ X1 / obs_var
    cov_batch = jnp.linalg.inv(posterior_prec)
    b = jnp.linalg.inv(Sigma0) @ mu0 + X1.T @ y / obs_var
    mu_batch = jnp.linalg.solve(posterior_prec, b)
    return mu_batch, cov_batch

def compare_dynamax_to_batch():
    params, X, Y = make_params_and_data(100, 2)
    nfeatures = params['F'].shape[0]
    lgssm = LinearGaussianSSM(state_dim = nfeatures, emission_dim = 1, input_dim = 0)

    params_lgssm, _ = lgssm.initialize(
        initial_mean=params['mu0'],
        initial_covariance=params['Sigma0'],
        dynamics_weights=params['F'],
        dynamics_covariance=params['Q'],
        emission_weights=params['Ht'],
        emission_covariance=params['R']
        )

    lgssm_posterior = lgssm.filter(params_lgssm, Y) 

    mu_kf = lgssm_posterior.filtered_means[-1]
    cov_kf = lgssm_posterior.filtered_covariances[-1]
    mu_batch, cov_batch = batch_bayes(X,Y)
    assert allclose(mu_batch, mu_kf)
    assert allclose(cov_batch, cov_kf)

    return lgssm_posterior

def compare_jax_to_dynamax():
    lgssm_posterior = compare_dynamax_to_batch()
    params, X, Y = make_params_and_data(100, 2)
    return_covs = True
    ll, kf_means, kf_covs = kf(params, Y, return_covs) 
    assert allclose(ll, lgssm_posterior.marginal_loglik)
    assert allclose(kf_means, lgssm_posterior.filtered_means)
    if return_covs:
        assert allclose(kf_covs, lgssm_posterior.filtered_covariances)

def time_jax(N, D):
    param_dict, X, Y = make_params_and_data(N, D)
    t0 = time.time()
    _ = kf(param_dict, Y, return_covs=False) 
    return time.time() - t0

### TORCH

def predict_pt(m, S, F, Q):
    mu_pred = F @ m 
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred

def condition_on_pt(m, P, H, R, y):
    S = R + H @ P @ H.T
    K = torch.linalg.solve(S + 1e-6, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - H @ m)
    return mu_cond, Sigma_cond

def kf_pt(params, emissions, return_covs=False, compile=False):
    F, Q, R = params['F'], params['Q'], params['R']
    def step(carry, t):
        ll, pred_mean, pred_cov = carry
        H = params['Ht'][t]
        y = emissions[t]
        #ll += MVN(H @ pred_mean, H @ pred_cov @ H.T + R).log_prob(y)
        filtered_mean, filtered_cov = condition_on_pt(pred_mean, pred_cov, H, R, y)
        pred_mean, pred_cov = predict_pt(filtered_mean, filtered_cov, F, Q)
        carry = (ll, pred_mean, pred_cov)
        if return_covs:
            return carry, (filtered_mean, filtered_cov)
        else:
            return carry, filtered_mean
    
    if compile:
        if platform.system() == 'Darwin':
            # https://discuss.pytorch.org/t/torch-compile-seems-to-hang/177089
            step = torch.compile(step, backend="aot_eager")
        else:
            step = torch.compile(step)
    num_timesteps = len(emissions)
    D = len(params['mu0'])
    filtered_means = torch.zeros((num_timesteps, D))
    if return_covs:
        filtered_covs = torch.zeros((num_timesteps, D, D))
    else:
        filtered_covs = None
    ll = 0
    carry = (ll, params['mu0'], params['Sigma0'])
    for t in range(num_timesteps):
        if return_covs:
            carry, (filtered_means[t], filtered_covs[t]) = step(carry, t)
        else:
            carry, filtered_means[t] = step(carry, t)
    return ll, filtered_means, filtered_covs

## Compare jax and torch

def convert_params_to_pt(params, Y):
    F, Q, R, Ht, mu0, Sigma0 = params['F'], params['Q'], params['R'], params['Ht'], params['mu0'], params['Sigma0']
    Y_pt = torch.tensor(np.array(Y))
    F_pt = torch.tensor(np.array(F))
    Q_pt = torch.tensor(np.array(Q))
    R_pt = torch.tensor(np.array(R))
    Ht_pt = torch.tensor(np.array(Ht))
    mu0_pt = torch.tensor(np.array(mu0))
    Sigma0_pt = torch.tensor(np.array(Sigma0))
    param_dict_pt = {'mu0': mu0_pt, 'Sigma0': Sigma0_pt, 'F': F_pt, 'Q': Q_pt, 'R': R_pt, 'Ht': Ht_pt}
    return param_dict_pt, Y_pt

def make_linreg_data_pt(N, D):
    torch.manual_seed(0)
    X = torch.randn((N, D))
    w = torch.randn((D, 1))
    y = X @ w + 0.1*torch.randn((N, 1))
    return X, y

def make_linreg_data_np(N, D):
    np.random.seed(0)
    X = np.random.randn(N, D)
    w = np.random.randn(D, 1)
    y = X @ w + 0.1*np.random.randn(N, 1)
    return X, y

def make_params_and_data_pt(N, D):
    X_np, Y_np = make_linreg_data_np(N, D)
    N, D = X_np.shape
    X1_np = np.column_stack((np.ones(N), X_np))  # Include column of 1s
    Ht_np = X1_np[:, None, :] # (T,D) -> (T,1,D), yhat = H[t]'z = (b w)' (1 x)
    nfeatures = X1_np.shape[1] # D+1
    Ht_pt = torch.tensor(Ht_np, dtype=torch.double)
    mu0_pt = torch.zeros(nfeatures, dtype=torch.double)
    Sigma0_pt = torch.eye(nfeatures, dtype=torch.double) * 1
    F_pt = torch.eye(nfeatures, dtype=torch.double) # dynamics = I
    Q_pt = torch.zeros((nfeatures, nfeatures), dtype=torch.double)  # No parameter drift.
    R_pt = torch.ones((1, 1), dtype=torch.double) * 0.1
    Y_pt = torch.tensor(Y_np)
    param_dict_pt = {'mu0': mu0_pt, 'Sigma0': Sigma0_pt, 'F': F_pt, 'Q': Q_pt, 'R': R_pt, 'Ht': Ht_pt}
    return param_dict_pt, X_np, Y_pt

def compare_torch_to_jax():
    print('compare_torch_to_jax')
    params, X, Y = make_params_and_data(100, 2)
    return_covs = True
    ll, kf_means, kf_covs = kf(params, Y, return_covs) 

    param_dict_pt, Y_pt = convert_params_to_pt(params, Y)
    ll_pt, kf_means_pt, kf_covs_pt = kf_pt(param_dict_pt, Y_pt, return_covs, compile=False) 
    assert(allclose(kf_means, kf_means_pt))
    if return_covs:
        assert(allclose(kf_covs, kf_covs_pt))

def time_torch(N, D, compile=False):
    param_dict_pt, X, Y_pt = make_params_and_data_pt(N, D)
    return_covs = False
    t0 = time.time()
    _ = kf_pt(param_dict_pt, Y_pt, return_covs, compile=compile) 
    return time.time() - t0

def main():
    #torch_only = False
    #if torch_only:
    #    dev = xm.xla_device()
    #    print('using device {}'.format(dev))
    ##    compile = False
    #    runtime_torch = time_torch(N, D, compile=compile)
    #    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))
    #    return
    
    try:
        tpus = jax.devices('tpu')
        dev = tpus[0]
    except:
        dev = jax.devices('cpu')[0]
    print('using device {}'.format(dev))
    
    compare_dynamax_to_batch()    
    compare_jax_to_dynamax()
    compare_torch_to_jax()
    N = 100
    D = 500
    runtime_jax = time_jax(N, D)
    print('jax, time={:.3f} N {} D {}'.format(runtime_jax, N, D))

    compile = False
    runtime_torch = time_torch(N, D, compile=compile)
    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))

    compile = True
    runtime_torch = time_torch(N, D, compile=compile)
    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))

if __name__ == '__main__':
    main()