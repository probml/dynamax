# Demo showcasing the online training of a binary logistic regression model
# using Conditional-Moments Gaussian Filter (CMGF).
# Authors: Peter G. Chang (@petergchang) and Gerardo Durán-Martín (@gerdm)

import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.optimize import minimize

from dynamax.cond_moments_gaussian_filter.containers import EKFParams, UKFParams, GHKFParams
from dynamax.cond_moments_gaussian_filter.cmgf import conditional_moments_gaussian_filter

def plot_posterior_predictive(ax, X, title, colors, Xspace=None, Zspace=None, cmap="viridis"):
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
        ax.contourf(*Xspace, Zspace, cmap=cmap, levels=20)
        ax.axis('off')
    ax.scatter(*X.T, c=colors, edgecolors='gray', s=50)
    ax.set_title(title)
    plt.tight_layout()


def plot_cmgf_post_laplace(mean_hist, cov_hist, w_laplace, filter_type, lcolors=["black", "tab:blue", "tab:red"]):
    """Plot convergence of posterior mean of CMGF estimation of parameters to
    Laplace posterior solution.

    Args:
        mean_hist (DeviceArray): Time series of CMGF posterior estimation of parameters at each step.
        cov_hist (DeviceArray): Time series of covariances of CMGF posterior estimation at each step.
        w_laplace (DeviceArray): Laplace posterior estimate of parameters.
        filter_type (str): Type of CMGF, for labelling purposes.
        lcolors (list, optional): _description_. Defaults to ["black", "tab:blue", "tab:red"].
    """    
    legend_font_size = 14
    bb1 = (1.1, 1.1)
    bb2 = (1.1, 0.3)
    bb3 = (0.8, 0.3)
    input_dim = mean_hist.shape[-1]
    tau_hist = jnp.array([cov_hist[:, i, i] for i in range(input_dim)]).T
    elements = (mean_hist.T, tau_hist.T, w_laplace, lcolors)
    n_datapoints = len(mean_hist)
    timesteps = jnp.arange(n_datapoints) + 1

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
    for k, (wk, Pk, wk_fix, c) in enumerate(zip(*elements)):
        axes[k].errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online ({filter_type})")
        axes[k].axhline(y=wk_fix, c=c, linestyle="dotted", label=f"$w_{k}$ batch (Laplace)", linewidth=3)
        axes[k].set_xlim(1, n_datapoints)
        axes[k].set_xlabel("ordered sample number", fontsize=15)
        axes[k].set_ylabel("weight value", fontsize=15)
        axes[k].tick_params(axis="both", which="major", labelsize=15)
        sns.despine()
        if k == 0:
            axes[k].legend(frameon=False, loc="upper right", bbox_to_anchor=bb1, fontsize=legend_font_size)

        elif k == 1:
            axes[k].legend(frameon=False, bbox_to_anchor=bb2, fontsize=legend_font_size)

        elif k == 2:
            axes[k].legend(frameon=False, bbox_to_anchor=bb3, fontsize=legend_font_size)
    plt.tight_layout()
    return fig


def generate_2d_binary_classification_data(num_points=1000, shuffle=True, key=0):
    """Generate balanced, standardized 2d binary classification dataset.

    Args:
        num_points (int, optional): Number of points to generate. Defaults to 1000.
        shuffle (bool, optional): Flag to determine whether to return shuffled dataset. Defaults to True.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.

    Returns:
        input: Generated input.
        input_with_bias: Generated input with bias term attached.
        output: Generated binary output.
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key0, key1, key2 = jr.split(key, 3)

    # Generate standardized noisy inputs that correspond to output '0'
    num_zero_points = num_points // 2
    zero_input = jnp.array([[-1., -1.]] * num_zero_points)
    zero_input += jr.normal(key0, (num_zero_points, 2))

    # Generate standardized noisy inputs that correspond to output '1'
    num_one_points = num_points - num_zero_points
    one_input = jnp.array([[1., 1.]] * num_one_points)
    one_input += jr.normal(key1, (num_one_points, 2))

    # Stack the inputs and add bias term
    input = jnp.concatenate([zero_input, one_input])
    input_with_bias = jnp.concatenate([jnp.ones((num_points, 1)), input], axis=1)

    # Generate binary output
    output = jnp.concatenate([jnp.zeros((num_zero_points)), jnp.ones((num_one_points))])

    # Shuffle
    if shuffle:
        idx = jr.permutation(key2, jnp.arange(num_points))
        input, input_with_bias, output = input[idx], input_with_bias[idx], output[idx]
    
    return input, input_with_bias, output


def generate_input_grid(input):
    """Generate grid on input space.

    Args:
        input (DeviceArray): Input array to determine the range of the grid.

    Returns:
        input_grid: Generated input grid.
        input_with_bias_grid: Generated input grid with bias term attached.
    """    
    # Define grid limits
    xmin, ymin = input.min(axis=0) - 0.1
    xmax, ymax = input.max(axis=0) + 0.1

    # Define grid
    step = 0.1
    input_grid = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = input_grid.shape
    input_with_bias_grid = jnp.concatenate([jnp.ones((1, nx, ny)), input_grid])

    return input_grid, input_with_bias_grid


def posterior_predictive_grid(grid, mean, cov, n_samples=5000, key=0):
    """Compute posterior predictive probability for each point in grid

    Args:
        grid (DeviceArray): Grid on which to predict posterior probability.
        mean (DeviceArray): Posterior mean of parameters.
        cov (DeviceArray): Covariance of posterior.
        n_samples (int, optional): Number of samples to use for prediction. Defaults to 5000.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    samples = jax.random.multivariate_normal(key, mean, cov, (n_samples,))
    Z = jax.nn.sigmoid(jnp.einsum("mij,sm->sij", grid, samples))
    Z = Z.mean(axis=0)
    return Z


def laplace_inference(X, Y, prior_var=2.0, key=0):
    """Perform Laplace inference and return posterior.

    Args:
        X (DeviceArray): Input array.
        Y (DeviceArray): Output label array.
        prior_var (float, optional): Prior variance for regularization term. Defaults to 2.0.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.

    Returns:
        w_laplace: Laplace posterior estimate of parameters.
        cov_laplace: Covariance of Laplace posterior estimation.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    input_dim = X.shape[-1]

    def log_posterior(w, X, Y, prior_var):
        prediction = jax.nn.sigmoid(X @ w)
        log_prior = -(prior_var * w @ w / 2)
        log_likelihood = Y * jnp.log(prediction) + (1 - Y) * jnp.log(1 - prediction)
        return log_prior + log_likelihood.sum()

    # Initial random guess
    w0 = jr.multivariate_normal(key, jnp.zeros(input_dim), jnp.eye(input_dim) * prior_var)
    
    # Energy function to minimize
    E = lambda w: -log_posterior(w, X, Y, prior_var) / len(Y)

    # Minimize energy function
    w_laplace = minimize(E, w0, method="BFGS").x
    cov_laplace = jax.hessian(E)(w_laplace)

    return w_laplace, cov_laplace


def generate_cmgf_params(Params, input_with_bias, prior_var=2.0):
    """Generate CMGF object with default initial parameters.

    Args:
        Params (CMGFParams): CMGFParams object to instantiate.
        input_with_bias (DeviceArray): Input array with bias term.
        prior_var (float, optional): Prior variance. Defaults to 2.0.

    Returns:
        cmgf_params: CMGFParams instance.
    """    
    input_dim = input_with_bias.shape[-1]
    sigmoid_fn = lambda w, x: jax.nn.sigmoid(w @ x)

    # Initial parameters for all CMGF methods
    initial_mean, initial_covariance = jnp.zeros(input_dim), prior_var * jnp.eye(input_dim)
    dynamics_function = lambda w, x: w
    dynamics_covariance = jnp.zeros((input_dim, input_dim))
    emission_mean_function = sigmoid_fn
    emission_cov_function = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))

    cmgf_params = Params(
        initial_mean = initial_mean,
        initial_covariance = initial_covariance,
        dynamics_function = dynamics_function,
        dynamics_covariance = dynamics_covariance,
        emission_mean_function = emission_mean_function,
        emission_cov_function = emission_cov_function
    )
    
    return cmgf_params


def main():
    # Generate 2d binary classification dataset
    input, input_with_bias, output = generate_2d_binary_classification_data()

    # Generate grid on input space
    input_grid, input_with_bias_grid = generate_input_grid(input)

    # Compute Laplace posterior
    prior_var = 1.0
    w_lap, cov_lap = laplace_inference(input_with_bias, output, prior_var=prior_var)

    # Compute CMGF-EKF, CMGF-UKF, and CMGF-GHKF posteriors
    cmgf_params = [generate_cmgf_params(params, input_with_bias, prior_var)
                   for params in [EKFParams, UKFParams, GHKFParams]]
    cmgf_posts = [conditional_moments_gaussian_filter(params, output, inputs=input_with_bias)
                  for params in cmgf_params]
    cmgf_means = [post.filtered_means for post in cmgf_posts]
    cmgf_covs = [post.filtered_covariances for post in cmgf_posts]

    all_figures = {}

    # Plot Laplace posterior predictive distribution
    fig, ax = plt.subplots()
    Z_lap = posterior_predictive_grid(input_with_bias_grid, w_lap, cov_lap)
    title_lap = "Laplace Predictive Distribution"
    colors = ['black' if y else 'red' for y in output]
    plot_posterior_predictive(ax, input, title_lap, colors, input_grid, Z_lap)
    all_figures['laplace_pred_dist'] = fig

    # Plot CMGF posterior predictive distributions
    cmgf_types = ['EKF', 'UKF', 'GHKF']
    for i, _ in enumerate(cmgf_means):
        fig, ax = plt.subplots()
        Z_cmgf = posterior_predictive_grid(input_with_bias_grid, cmgf_means[i][-1], cmgf_covs[i][-1])
        title_cmgf = f'CMGF-{cmgf_types[i]} Predictive Distribution'
        plot_posterior_predictive(ax, input, title_cmgf, colors, input_grid, Z_cmgf)
        all_figures[f'cmgf_{cmgf_types[i].lower()}_pred_dist'] = fig

    # Plot convergence of CMGF posteriors to Laplace posterior
    for i, _ in enumerate(cmgf_means):
        all_figures[f'cmgf_{cmgf_types[i].lower()}_lap_convergence'] = \
            plot_cmgf_post_laplace(cmgf_means[i][::max(1, len(output)//100)],
                                   cmgf_covs[i][::max(1, len(output)//100)],
                                   w_lap, f'CMGF-{cmgf_types[i]}')

    return all_figures


if __name__ == '__main__':
    figures = main()
    plt.show()