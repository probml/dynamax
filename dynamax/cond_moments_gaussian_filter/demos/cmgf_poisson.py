# Demo showcasing the online training of a Poisson model
# using Conditional-Moments Gaussian Filter (CMGF).
# Authors: Peter G. Chang (@petergchang) and Collin Schlager (@schlagercollin)

import warnings

from functools import partial
import matplotlib.pyplot as plt
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import Poisson as Pois
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jax.tree_util import tree_map

from dynamax.cond_moments_gaussian_filter.cmgf import conditional_moments_gaussian_smoother, EKFIntegrals, GHKFIntegrals
from dynamax.cond_moments_gaussian_filter.generalized_gaussian_ssm import GGSSM, GGSSMParams


def plot_states(states, num_steps, title, ax):
    """Plot latent states.

    Args:
        states (DeviceArray): Time series data of latent states to plot.
        num_steps (int): Number of steps of states to plot.
        title (str): Title of the plot.
        ax (axis): Matplotlib axis.
    """    
    latent_dim = states.shape[-1]
    lim = abs(states).max()
    for d in range(latent_dim):
        ax.plot(states[:, d] + lim * d, "-")
    ax.set_yticks(jnp.arange(latent_dim) * lim, ["$z_{}$".format(d + 1) for d in range(latent_dim)])
    ax.set_xticks([])
    ax.set_xlim(0, num_steps)
    ax.set_title(title)
    return ax


def plot_emissions_poisson(states, data):
    """Plot batches of samples generated via Poisson likelihood.

    Args:
        states (DeviceArray): Latent states.
        data (DeviceArray): Observations.
    """    
    latent_dim = states.shape[-1]
    emissions_dim = data.shape[-1]
    num_steps = data.shape[0]

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        gridspec_kw={'height_ratios': [1, emissions_dim / latent_dim]}
    )

    # Plot the continuous latent states
    lim = abs(states).max()
    for d in range(latent_dim):
        axes[0].plot(states[:, d] + lim * d, "-")
    axes[0].set_yticks(jnp.arange(latent_dim) * lim, ["$z_{}$".format(d + 1) for d in range(latent_dim)])
    axes[0].set_xticks([])
    axes[0].set_xlim(0, num_steps)
    axes[0].set_title("Sampled Latent States")

    lim = abs(data).max()
    im = axes[1].imshow(data.T, aspect="auto", interpolation="none")
    axes[1].set_xlabel("time")
    axes[1].set_xlim(0, num_steps)
    axes[1].set_yticks(ticks=jnp.arange(emissions_dim))
    axes[1].set_ylabel("Emission dimension")
    axes[1].set_title("Sampled Emissions (Counts / Time Bin)")
    plt.colorbar(im, ax=axes[1])
    return fig


def compare_dynamics(Ex, states, data, dynamics_weights, dynamics_bias, filter_type=''):
    """Compare dynamics of states between posterior and ground truth values.

    Args:
        Ex (DeviceArray): Posterior means of latent states.
        states (DeviceArray): Ground truth of latent states.
        data (DeviceArray): Observations.
        dynamics_weights (DeviceArray): Dynamics weights.
        dynamics_bias (DeviceArray): Dynamics bias.
    """    
    def plot_dynamics_2d(dynamics_matrix, bias_vector, mins=(-40,-40), maxs=(40,40), npts=20, axis=None, **kwargs):
        assert dynamics_matrix.shape == (2, 2), "Must pass a 2 x 2 dynamics matrix to visualize."
        assert len(bias_vector) == 2, "Bias vector must have length 2."

        x_grid, y_grid = jnp.meshgrid(jnp.linspace(mins[0], maxs[0], npts), jnp.linspace(mins[1], maxs[1], npts))
        xy_grid = jnp.column_stack((x_grid.ravel(), y_grid.ravel(), jnp.zeros((npts**2,0))))
        dx = xy_grid.dot(dynamics_matrix.T) + bias_vector - xy_grid

        if axis is not None:
            q = axis.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
        else:
            q = plt.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)

        plt.gca().set_aspect(1.0)
        return q

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    q = plot_dynamics_2d(
        dynamics_weights,
        dynamics_bias,
        mins=states.min(axis=0),
        maxs=states.max(axis=0),
        color="blue",
        axis=axes[0],
    )
    axes[0].plot(states[:, 0], states[:, 1], lw=2)
    axes[0].plot(states[0, 0], states[0, 1], "*r", markersize=10, label="$z_{init}$")
    axes[0].set_xlabel("$z_1$")
    axes[0].set_ylabel("$z_2$")
    axes[0].set_title("True Latent States & Dynamics")

    q = plot_dynamics_2d(
        dynamics_weights,
        dynamics_bias,
        mins=Ex.min(axis=0),
        maxs=Ex.max(axis=0),
        color="red",
        axis=axes[1],
    )

    axes[1].plot(Ex[:, 0], Ex[:, 1], lw=2)
    axes[1].plot(Ex[0, 0], Ex[0, 1], "*r", markersize=10, label="$z_{init}$")
    axes[1].set_xlabel("$z_1$")
    axes[1].set_ylabel("$z_2$")
    axes[1].set_title(f"{filter_type} Inferred Latent States & Dynamics")
    plt.tight_layout()
    return fig


def compare_smoothed_predictions(Ey, Ey_true, Covy, data, filter_type=''):
    """Compare smoothed predictions between posterior and ground truth values.

    Args:
        Ey (DeviceArray): Predicted observations using smoothed means.
        Ey_true (DeviceArray): Predicted observations using ground truth latent states.
        Covy (DeviceArray): Emission covariance.
        data (DeviceArray): Ground truth observations.
    """    
    data_dim = data.shape[-1]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(Ey_true + 10 * jnp.arange(data_dim))
    ax.plot(Ey + 10 * jnp.arange(data_dim), "--k")
    for i in range(data_dim):
        ax.fill_between(
            jnp.arange(len(data)),
            10 * i + Ey[:, i] - 2 * jnp.sqrt(Covy[:, i, i]),
            10 * i + Ey[:, i] + 2 * jnp.sqrt(Covy[:, i, i]),
            color="k",
            alpha=0.25,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("data and predictions (for each neuron)")
    ax.set_title(f'Comparison between {filter_type} smoothed posterior prediction and ground truth.')

    ax.plot([0], "--k", label="Predicted")  # dummy trace for legend
    ax.plot([0], "-k", label="True")
    ax.legend(loc="upper right")
    return fig


def random_rotation(dim, key=0, theta=None):
    """Compute the weight of dynamics that corresponds to a random rotation of theta.

    Args:
        dim (int): Dimension of states that undergo rotation.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.
        theta (_type_, optional): Angle of rotation. Defaults to None.

    Returns:
        w (DeviceArray): Dynamics weight of random rotation.
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    key1, key2 = jr.split(key)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * jnp.pi * jr.uniform(key1)

    if dim == 1:
        return jr.uniform(key1) * jnp.eye(1)

    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(dim)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key2, shape=(dim, dim)))[0]
    w = q.dot(out).dot(q.T)
    return w


def sample_poisson(params, poisson_weights, num_steps, num_trials, key=0):
    """Sample states and emissions using Poisson likelihood. The states are
    evolved according to the dynamics specified in params.

    Args:
        params (CMGFParams): CMGFParams object.
        poisson_weights (DeviceArray): Poisson weights.
        num_steps (int): Number of states/emissions to sample.
        num_trials (int): Number of batches to sample.
        key (int, optional): Initial PRNG seed for jax.random. Defaults to 0.

    Returns:
        states (DeviceArray): Sampled states.
        emissions (DeviceArray): Observations using Poisson likelihood on sampled states.
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    def _sample(key):
        key1, key2, subkey = jr.split(key, 3)
        initial_state = MVN(params.initial_mean, params.initial_covariance).sample(seed=key1)
        initial_emission = Pois(log_rate=poisson_weights @ initial_state).sample(seed=key2)

        def _step(carry, key):
            key1, key2 = jr.split(key, 2)
            prev_state = carry
            next_state = MVN(params.dynamics_function(prev_state), params.dynamics_covariance).sample(seed=key1)
            emission = Pois(log_rate=poisson_weights @ next_state).sample(seed=key2)

            return next_state, (next_state, emission)
        
        keys = jr.split(subkey, num_steps-1)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)

        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, states)
        emissions = tree_map(expand_and_cat, initial_emission, emissions)
        return states, emissions
    
    if num_trials > 1:
        batch_keys = jr.split(key, num_trials)
        states, emissions = vmap(_sample)(batch_keys)
    else:
        states, emissions = _sample(key)
        
    return states, emissions


def main():
    # Parameters for our Poisson demo
    state_dim, emission_dim = 2, 5
    poisson_weights = jr.normal(jr.PRNGKey(0), shape=(emission_dim, state_dim))

    # Construct CMGF parameters
    cmgf_params = GGSSMParams(
        initial_mean = jnp.zeros(state_dim),
        initial_covariance = jnp.eye(state_dim),
        dynamics_function = lambda z: random_rotation(state_dim, theta=jnp.pi/20) @ z,
        dynamics_covariance = 0.001 * jnp.eye(state_dim),
        emission_mean_function = lambda z: jnp.exp(poisson_weights @ z),
        emission_cov_function = lambda z: jnp.diag(jnp.exp(poisson_weights @ z)),
        emission_dist = lambda mu, _: Pois(log_rate = jnp.log(mu))
    )

    # Sample from random-rotation state dynamics and Poisson emissions
    num_steps, num_trials = 200, 3
    model = GGSSM(state_dim, emission_dim)
    sample_poisson = lambda key: model.sample(params=cmgf_params, num_timesteps=num_steps, key=key)
    keys = jr.split(jr.PRNGKey(0), num_trials)
    all_states, all_emissions = vmap(sample_poisson)(keys)

    figs = {}
    # Plot batches of samples generated
    figs['samples'] = plot_emissions_poisson(all_states[0], all_emissions[0])

    # Perform CMGF-Smoother Inference
    for filter_type, inf_params in {"CMGF-EKF": EKFIntegrals(), "CMGF-GHKF": GHKFIntegrals()}.items():
        posts = vmap(conditional_moments_gaussian_smoother, (None, None, 0))(cmgf_params, inf_params, all_emissions)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        plot_states(posts.smoothed_means[0], num_steps, f"{filter_type}-Inferred Latent States", ax)
        figs[f'{filter_type.lower()}_latent_states'] = fig

        for i in range(num_trials):
            fig_dyn = compare_dynamics(posts.smoothed_means[i], all_states[i], all_emissions[i],
                                random_rotation(state_dim, theta=jnp.pi/20), jnp.zeros(state_dim), filter_type)
            figs[f'{filter_type.lower()}_dynamics_comp_trial_{i}'] = fig_dyn

            fig_pred = compare_smoothed_predictions(
                posts.smoothed_means[i] @ poisson_weights.T,
                all_states[i] @ poisson_weights.T,
                poisson_weights @ posts.smoothed_covariances[i] @ poisson_weights.T,
                all_emissions[i],
                filter_type
            )
            figs[f'{filter_type.lower()}_pred_comp_trial_{i}'] = fig_pred

    return figs

if __name__ == '__main__':
    with warnings.catch_warnings():
        # Ignore tfp warnings
        warnings.simplefilter("ignore")
        figures = main()
        plt.show()
