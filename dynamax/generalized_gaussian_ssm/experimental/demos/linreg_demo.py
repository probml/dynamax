import chex
from typing import Callable, Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad
from jax.tree_util import tree_map, tree_reduce
import flax
import flax.linen as nn
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import chex
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jax.flatten_util import ravel_pytree

from dynamax.generalized_gaussian_ssm.inference import conditional_moments_gaussian_filter, EKFIntegrals
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM

from dynamax.generalized_gaussian_ssm.experimental.utils import *
from dynamax.generalized_gaussian_ssm.experimental.diagonal_inference import *

def generate_linreg_dataset(num_points=100, theta=jnp.array([1, 0.5]), var=0.1, key=1, shuffle=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    X = jnp.linspace(0.0, 1.0, num_points)
    Y_tr = theta[1]*X + theta[0]
    noise = var*jr.normal(key, shape=(num_points,))
    Y = Y_tr + noise
    X_sh, Y_sh = X, Y
    # Shuffle data
    if shuffle:
        shuffle_idx = jr.permutation(subkey, jnp.arange(num_points))
        X_sh, Y_sh = X[shuffle_idx], Y[shuffle_idx]

    return X, X_sh, Y, Y_sh, Y_tr

def main():
    # 100 data points
    X_lr_100, X_lr_100_sh, Y_lr_100, Y_lr_100_sh, Y_lr_100_tr = generate_linreg_dataset()

    # 200 data points
    X_lr_200, X_lr_200_sh, Y_lr_200, Y_lr_200_sh, Y_lr_200_tr = generate_linreg_dataset(200)

    plt.figure()
    plt.scatter(X_lr_100, Y_lr_100)
    plt.plot(X_lr_100, Y_lr_100_tr, c='red');
    plt.title('training data')

    # Define Linear Regression as single layer perceptron
    input_dim_lr, hidden_dims_lr, output_dim_lr = 1, [], 1
    model_dims_lr = [input_dim_lr, *hidden_dims_lr, output_dim_lr]
    _, flat_params_lr, _, apply_fn_lr = get_mlp_flattened_params(model_dims_lr)
    print('nparams ', flat_params_lr.shape)

    # Full covariariance  EKF for linear regression
    state_dim_lr, emission_dim_lr = flat_params_lr.size, output_dim_lr
    var_lr = 0.1
    fcekf_params_lr = ParamsGGSSM(
        initial_mean=flat_params_lr,
        initial_covariance=jnp.eye(state_dim_lr),
        dynamics_function=lambda w, _: w,
        dynamics_covariance = jnp.eye(state_dim_lr) * 0,
        emission_mean_function = lambda w, x: apply_fn_lr(w, x),
        emission_cov_function = lambda w, x: var_lr
    )
    v_apply_fn_lr = vmap(apply_fn_lr, (None, 0))

    # 100 datapoints
    fcekf_post_lr_100 = conditional_moments_gaussian_filter(fcekf_params_lr, EKFIntegrals(), Y_lr_100_sh, inputs=X_lr_100_sh)
    fcekf_theta_lr_100 = fcekf_post_lr_100.filtered_means[-1]
    print(f'fcekf_theta_lr_100 = {fcekf_theta_lr_100}')
    plt.figure()
    plt.scatter(X_lr_100, Y_lr_100)
    plt.plot(X_lr_100, v_apply_fn_lr(fcekf_theta_lr_100, X_lr_100), c='red');
    plt.title('full covariance N=100')

    # 200 datapoints
    fcekf_post_lr_200 = conditional_moments_gaussian_filter(fcekf_params_lr, EKFIntegrals(), Y_lr_200_sh, inputs=X_lr_200_sh)
    fcekf_theta_lr_200 = fcekf_post_lr_200.filtered_means[-1]
    print(f'fcekf_theta_lr_200 = {fcekf_theta_lr_200}')
    plt.figure()
    plt.scatter(X_lr_200, Y_lr_200)
    plt.plot(X_lr_200, v_apply_fn_lr(fcekf_theta_lr_200, X_lr_200), c='red');
    plt.title('full covariance N=200')

    # fully decoupled diagonal EKF for linear regression
    dekf_params_lr = DEKFParams(
        initial_mean=flat_params_lr,
        initial_cov_diag=jnp.ones((state_dim_lr,)),
        dynamics_cov_diag=jnp.ones((state_dim_lr,)) * 0,
        emission_mean_function = lambda w, x: apply_fn_lr(w, x),
        emission_cov_function = lambda w, x: var_lr
    )

    # 100 datapoints
    fdekf_post_lr_100 = stationary_dynamics_fully_decoupled_conditional_moments_gaussian_filter(dekf_params_lr, Y_lr_100, inputs=X_lr_100)
    fdekf_theta_lr_100 = fdekf_post_lr_100.filtered_means[-1]
    print(f'fdekf_theta_lr_100 = {fdekf_theta_lr_100}')
    plt.figure()
    plt.scatter(X_lr_100, Y_lr_100)
    plt.plot(X_lr_100, v_apply_fn_lr(fdekf_theta_lr_100, X_lr_100), c='red');
    plt.title('fully decoupled N=100')

    # 200 datapoints
    fdekf_post_lr_200 = stationary_dynamics_fully_decoupled_conditional_moments_gaussian_filter(dekf_params_lr, Y_lr_200, inputs=X_lr_200)
    fdekf_theta_lr_200 = fdekf_post_lr_200.filtered_means[-1]
    print(f'fdekf_theta_lr_200 = {fdekf_theta_lr_200}')
    plt.figure()
    plt.scatter(X_lr_200, Y_lr_200)
    plt.plot(X_lr_200, v_apply_fn_lr(fdekf_theta_lr_200, X_lr_200), c='red');
    plt.title('fully decoupled N=200')

    # SGD

    def loss_optax(params, x, y, loss_fn, apply_fn):
        y, y_hat = jnp.atleast_1d(y), apply_fn(params, x)
        loss_value = loss_fn(y, y_hat)
        return loss_value.mean()

    sgd_optimizer = optax.sgd(learning_rate=1e-2)
    # L2 loss function for linear regression
    loss_fn_lr = partial(loss_optax, loss_fn = optax.l2_loss, apply_fn = apply_fn_lr)

    # SGD One epoch

    # 100 datapoints
    sgd_sp_theta_lr_100 = fit_optax(flat_params_lr, sgd_optimizer, X_lr_100, Y_lr_100, loss_fn_lr, num_epochs=1)
    print(f'sgd_sp_theta_lr_100 = {sgd_sp_theta_lr_100}')
    plt.figure()
    plt.scatter(X_lr_100, Y_lr_100)
    plt.plot(X_lr_100, v_apply_fn_lr(sgd_sp_theta_lr_100, X_lr_100), c='red');
    plt.title('SGD, epochs=1, N=100')

    # 200 datapoints
    sgd_sp_theta_lr_200 = fit_optax(flat_params_lr, sgd_optimizer, X_lr_200, Y_lr_200, loss_fn_lr, num_epochs=1)
    print(f'sgd_sp_theta_lr_100 = {sgd_sp_theta_lr_200}')
    plt.figure()
    plt.scatter(X_lr_200, Y_lr_200)
    plt.plot(X_lr_200, v_apply_fn_lr(sgd_sp_theta_lr_200, X_lr_200), c='red');
    plt.title('SGD, epochs=1, N=200')


    # SGD 200 epochs

    # 100 datapoints
    sgd_mp_theta_lr_100 = fit_optax(flat_params_lr, sgd_optimizer, X_lr_100, Y_lr_100, loss_fn_lr, num_epochs=200)
    print(f'sgd_mp_theta_lr_100 = {sgd_mp_theta_lr_100}')
    plt.figure()
    plt.scatter(X_lr_100, Y_lr_100)
    plt.plot(X_lr_100, v_apply_fn_lr(sgd_mp_theta_lr_100, X_lr_100), c='red');
    plt.title('SGD, epochs=200, N=100')

    # 200 datapoints
    sgd_mp_theta_lr_200 = fit_optax(flat_params_lr, sgd_optimizer, X_lr_200, Y_lr_200, loss_fn_lr, num_epochs=200)
    print(f'sgd_mp_theta_lr_200 = {sgd_mp_theta_lr_200}')
    plt.figure()
    plt.scatter(X_lr_200, Y_lr_200)
    plt.plot(X_lr_200, v_apply_fn_lr(sgd_mp_theta_lr_200, X_lr_200), c='red');
    plt.title('SGD, epochs=200, N=200')

    plt.show()

if __name__ == "__main__":
    main()