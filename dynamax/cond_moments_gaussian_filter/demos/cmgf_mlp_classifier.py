# Demo showcasing the online training of a MLP-Classifier
# using Conditional-Moments Gaussian Filter (CMGF).
# Author: Peter G. Chang (@petergchang)

from typing import Sequence
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import flax.linen as nn

from dynamax.cond_moments_gaussian_filter.cmgf import conditional_moments_gaussian_filter, EKFParams

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
        binary (bool, optional): Flag to determine whether to round probabilities to binary outputs. Defaults to True.

    Returns:
        _type_: _description_
    """    
    inferred_fn = lambda x: apply(mean, x)
    fn_vec = jnp.vectorize(inferred_fn, signature='(2)->(3)')
    Z = fn_vec(grid)
    if binary:
        Z = jnp.rint(Z)
    return Z


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def get_mlp_flattened_params(model_dims, key=0):
    """Generate MLP model, initialize it using dummy input, and
    return the model, its flattened initial parameters, function
    to unflatten parameters, and apply function for the model.

    Args:
        model_dims (List): List of [input_dim, hidden_dim, ..., output_dim]
        key (PRNGKey): Random key. Defaults to 0.

    Returns:
        model: MLP model with given feature dimensions.
        flat_params: Flattened parameters initialized using dummy input.
        unflatten_fn: Function to unflatten parameters.
        apply_fn: fn(flat_params, x) that returns the result of applying the model.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    # Define MLP model
    input_dim, features = model_dims[0], model_dims[1:]
    model = MLP(features)
    dummy_input = jnp.ones((input_dim,))

    # Initialize parameters using dummy input
    params = model.init(key, dummy_input)
    flat_params, unflatten_fn = ravel_pytree(params)

    # Define apply function
    def apply(flat_params, x, model, unflatten_fn):
        return model.apply(unflatten_fn(flat_params), jnp.atleast_1d(x))

    apply_fn = partial(apply, model=model, unflatten_fn=unflatten_fn)

    return model, flat_params, unflatten_fn, apply_fn


def main():
    # Define MLP architecture
    input_dim, hidden_dims, output_dim = 2, [15, 15], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

    figs = {}
    # Generate spiral dataset and plot
    input, output = generate_spiral_dataset()
    fig, ax = plt.subplots(figsize=(6, 5))
    title = "Spiral-shaped binary classification data"
    plot_posterior_predictive(ax, input, output, title)
    figs['dataset'] = fig

    # Run CMGF-EKF to train the MLP Classifier
    state_dim, emission_dim = flat_params.size, output_dim
    sigmoid_fn = lambda w, x: jax.nn.sigmoid(apply_fn(w, x))
    cmgf_ekf_params = EKFParams(
        initial_mean=flat_params,
        initial_covariance=jnp.eye(state_dim),
        dynamics_function=lambda w, _: w,
        dynamics_covariance=jnp.eye(state_dim) * 1e-4,
        emission_mean_function = lambda w, x: sigmoid_fn(w, x),
        emission_cov_function = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))
    )
    cmgf_ekf_post = conditional_moments_gaussian_filter(cmgf_ekf_params, output, inputs=input)
    w_means, w_covs = cmgf_ekf_post.filtered_means, cmgf_ekf_post.filtered_covariances

    # Define grid on input space
    input_grid = generate_input_grid(input)

    # Evaluate the trained MLP on grid and plot
    Z_cmgf = posterior_predictive_grid(input_grid, w_means[-1], sigmoid_fn, binary=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    title = "CMGF-EKF One-Pass Trained MLP Classifier"
    plot_posterior_predictive(ax, input, output, title, input_grid, Z_cmgf)
    figs['cmgf_ekf_final'] = fig

    # Plot intermediate predictions
    intermediate_steps = [9, 49, 99, 199, 299, 399]
    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    for step, axi in zip(intermediate_steps, ax.flatten()):
        Zi = posterior_predictive_grid(input_grid, w_means[step], sigmoid_fn)
        title = f'step={step+1}'
        plot_posterior_predictive(axi, input[:step+1], output[:step+1], title, input_grid, Zi)
    plt.tight_layout()
    figs['cmgf_ekf_intermediate'] = fig

    # Save training as .mp4 video
    def animate(i):
        ax.cla()
        w_curr = w_means[i]
        Zi = posterior_predictive_grid(input_grid, w_means[i], sigmoid_fn)
        title = f'CMGF-EKF-MLP ({i+1}/500)'
        plot_posterior_predictive(ax, input[:i+1], output[:i+1], title, input_grid, Zi)
        return ax
    fig, ax = plt.subplots(figsize=(6, 5))
    anim = animation.FuncAnimation(fig, animate, frames=500, interval=50)
    anim.save("cmgf_mlp_classifier.mp4", dpi=200, bitrate=-1, fps=24)

    return figs

if __name__ == "__main__":
    figures = main()
    plt.show()
