from numpy import var
import numpy as np


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
from dynamax.rebayes.utils import *
from dynamax.rebayes.diagonal_inference import *
from dynamax.rebayes.ekf import RebayesEKF

def generate_input_grid(input):
    """Generate grid on input space.
    Args:
        input (DeviceArray): Input array to determine the range of the grid.
    Returns:
        input_grid: Generated input grid.
    """    
    # Define grid limits
    xmin, ymin = input.min(axis=0) - 0.1
    xmax, ymax = input.max(axis=0) + 0.1

    # Define grid
    step = 0.1
    x_grid, y_grid = jnp.meshgrid(jnp.mgrid[xmin:xmax:step], jnp.mgrid[ymin:ymax:step])
    input_grid = jnp.concatenate([x_grid[...,None], y_grid[...,None]], axis=2)

    return input_grid

def posterior_predictive_grid(grid, mean, apply, binary=False):
    """Compute posterior predictive probability for each point in grid
    Args:
        grid (DeviceArray): Grid on which to predict posterior probability.
        mean (DeviceArray): Posterior mean of parameters.
        apply (Callable): Apply function for MLP.
        binary (bool, optional): Flag to determine whether to round probabilities to binary outputs. Defaults to False.
    Returns:
        _type_: _description_
    """    
    inferred_fn = lambda x: apply(mean, x)
    fn_vec = jnp.vectorize(inferred_fn, signature='(2)->(3)')
    Z = fn_vec(grid)
    if binary:
        Z = jnp.rint(Z)
    return Z

def plot_posterior_predictive(ax, X, Y, title, Xspace=None, Zspace=None, cmap=cm.rainbow):
    """Plot the 2d posterior predictive distribution.
    Args:
        ax (axis): Matplotlib axis.
        X (DeviceArray): Input array.
        title (str): Title for the plot.
        colors (list): List of colors that correspond to each element in X.
        Xspace (DeviceArray, optional): Input grid to predict on. Defaults to None.
        Zspace (DeviceArray, optional): Predicted posterior on the input grid. Defaults to None.
        cmap (str, optional): Matplotlib colormap. Defaults to "viridis".
    """    
    if Xspace is not None and Zspace is not None:
        ax.contourf(*(Xspace.T), (Zspace.T[0]), cmap=cmap, levels=50)
        ax.axis('off')
    colors = ['red' if y else 'blue' for y in Y]
    ax.scatter(*X.T, c=colors, edgecolors='black', s=50)
    ax.set_title(title)
    plt.tight_layout()
    return ax

def generate_spiral_dataset(num_per_class=250, zero_var=1., one_var=1., shuffle=True, key=0):
    """Generate balanced, standardized 2d "spiral" binary classification dataset.
    Code adapted from https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
    Args:
        num_per_class (int, optional): Number of points to generate per class. Defaults to 250.
        zero_val (float, optional): Noise variance for inputs withj label '0'. Defaults to 1.
        one_val (float, optional): Noise variance for inputs withj label '1'. Defaults to 1.
        shuffle (bool, optional): Flag to determine whether to return shuffled dataset. Defaults to True.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.
    Returns:
        input: Generated input.
        output: Generated binary output.
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key1, key2, key3, key4 = jr.split(key, 4)

    theta = jnp.sqrt(jr.uniform(key1, shape=(num_per_class,))) * 2*jnp.pi
    r = 2*theta + jnp.pi
    generate_data = lambda theta, r: jnp.array([jnp.cos(theta)*r, jnp.sin(theta)*r]).T

    # Input data for output zero
    zero_input = generate_data(theta, r) + zero_var * jr.normal(key2, shape=(num_per_class, 2))

    # Input data for output one
    one_input = generate_data(theta, -r) + one_var * jr.normal(key3, shape=(num_per_class, 2))

    # Stack the inputs and standardize
    input = jnp.concatenate([zero_input, one_input])
    input = (input - input.mean(axis=0)) / input.std(axis=0)

    # Generate binary output
    output = jnp.concatenate([jnp.zeros(num_per_class), jnp.ones(num_per_class)])

    if shuffle:
        idx = jr.permutation(key4, jnp.arange(num_per_class * 2))
        input, output = input[idx], output[idx]

    return input, output
    
def make_data_and_model():
    num_per_class = 500 # 1000 data points total
    X, Y = generate_spiral_dataset(num_per_class)
    ntrain = 500
    data = {'Xtrain': X[:ntrain], 'Ytrain': Y[:ntrain], 'Xtest': X[ntrain:], 'Ytest': Y[ntrain:]}
    #fig, ax = plt.subplots()
    #plot_posterior_predictive(ax, X_nc, Y_nc, "Nonlinearly-Separable classification");

    # Define MLP architecture
    #input_dim_nc, hidden_dims_nc, output_dim_nc = 2, [30, 50], 1
    input_dim_nc, hidden_dims_nc, output_dim_nc = 2, [20, 20], 1
    model_dims_nc = [input_dim_nc, *hidden_dims_nc, output_dim_nc]
    _, flat_params, _, apply_fn_nc = get_mlp_flattened_params(model_dims_nc)
    print('nparams ', flat_params.shape)

    eps_nc = 1e-4
    predict_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(apply_fn_nc(w, x)), eps_nc, 1-eps_nc) # Clip to prevent divergence

    return data, flat_params, predict_fn

def misclassification_rate(Ytrue, Yprob): # binary labels
    Yhat = (Yprob[:,0] > 0.5)
    nerrors = jnp.sum(Yhat != Ytrue)
    nsamples = len(Ytrue)
    rate = nerrors*1.0 / nsamples
    return rate

def eval_perf(data, weights, predict_fn):
    inferred_fn = lambda x: predict_fn(weights, x)
    error_rate_train = misclassification_rate(data['Ytrain'], inferred_fn(data['Xtrain']))
    error_rate_test = misclassification_rate(data['Ytest'], inferred_fn(data['Xtest']))
    return error_rate_train, error_rate_test

def plot_learning_curves(ax, data, weights_history, predict_fn, title, xlabel='num data points'):
    fn = lambda w: misclassification_rate(data['Ytrain'], predict_fn(w, data['Xtrain']))
    error_rates_train = vmap(fn)(weights_history)
    fn = lambda w: misclassification_rate(data['Ytest'], predict_fn(w, data['Xtest']))
    error_rates_test = vmap(fn)(weights_history)

    #plt.figure(figsize=(5,5))
    ax.plot(error_rates_train, label='train')
    ax.plot(error_rates_test, label='test')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('misclassification rate')
    ax.set_title(title)

def plot_heatmap(data, weights_history, predict_fn, title):
    # Evaluate the trained MLP on grid and plot
    X_nc, Y_nc = data['Xtrain'], data['Ytrain']
    input_grid_nc = generate_input_grid(X_nc)
    Z = posterior_predictive_grid(input_grid_nc, weights_history[-1], predict_fn, binary=False)
    fig, ax = plt.subplots(figsize=(3,3))
    plot_posterior_predictive(ax, X_nc, Y_nc, title, input_grid_nc, Z);

    intermediate_steps = [10, 50, 100, len(Y_nc)]
    fig, ax = plt.subplots(2, 2, figsize=(6,6))
    for step, axi in zip(intermediate_steps, ax.flatten()):
        Zi = posterior_predictive_grid(input_grid_nc, weights_history[step-1], predict_fn, binary=False)
        title = f'step={step}'
        plot_posterior_predictive(axi, X_nc[:step], Y_nc[:step], title, input_grid_nc, Zi)
    plt.tight_layout()


def run_ekf(data, flat_params, predict_fn, type='fcekf'):
    X, Y = data['Xtrain'], data['Ytrain']
    ekf_params, callback = initialize_params(flat_params, predict_fn)
    estimator = RebayesEKF(ekf_params, method = type)
    _, filtered_means = estimator.scan(X, Y, callback=callback)

    return filtered_means


def sgd_old(data, flat_params, predict_fn):
    X_nc, Y_nc = data['Xtrain'], data['Ytrain']
    # Cross entropy loss for nonlinear classification
    loss_fn_nc = partial(loss_optax, loss_fn = lambda y, yhat: -(y * jnp.log(yhat) + (1-y) * jnp.log(1 - yhat)), 
                     apply_fn = predict_fn)
    epoch_range = jnp.array([1, 5, 10, 20])
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    input_grid_nc = generate_input_grid(X_nc)
    for step, axi in zip(epoch_range, ax.flatten()):
        params_step = fit_optax(flat_params, sgd_optimizer, X_nc, Y_nc, loss_fn_nc, num_epochs=step)

        error_rate_train, error_rate_test = eval_perf(data, params_step, predict_fn)
        print('nepochs {}, error rate train {:.3f}, test {:.3f}'.format(step, error_rate_train, error_rate_test))

        Zi = posterior_predictive_grid(input_grid_nc, params_step, predict_fn, binary=False)
        title = f'nepochs={step}'
        plot_posterior_predictive(axi, X_nc, Y_nc, title, input_grid_nc, Zi)
    plt.tight_layout()


def sgd(data, flat_params, predict_fn):
    X_nc, Y_nc = data['Xtrain'], data['Ytrain']
    params = flat_params
    # Cross entropy loss for nonlinear classification
    loss_fn = partial(loss_optax, loss_fn = lambda y, yhat: -(y * jnp.log(yhat) + (1-y) * jnp.log(1 - yhat)), 
                     apply_fn = predict_fn)
    optimizer = optax.sgd(learning_rate=1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    num_epochs = 20
    num_params = params.shape[0]
    weights_history = np.zeros((num_epochs, num_params))
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(X_nc, Y_nc)):
            params, opt_state, loss_value = step(params, opt_state, x, y)
        weights_history[epoch] = params
    
    return weights_history

def main():
    data, flat_params, predict_fn = make_data_and_model()
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axi = axs.flatten()

    ekf_types = {'fcekf': 'full cov', 'fdekf': 'fully decoupled', 'vdekf': 'variational diagonal'}
    for ekf_type, ax in zip(ekf_types.keys(), axi[:3]):
        weights_history = run_ekf(data, flat_params, predict_fn, type=ekf_type)
        plot_learning_curves(ax, data, weights_history, predict_fn, ekf_types[ekf_type])

    weights_history = sgd(data, flat_params, predict_fn)
    plot_learning_curves(axi[3], data, weights_history, predict_fn, 'SGD', xlabel='num epochs')

    plt.show()

if __name__ == "__main__":
    main()