import matplotlib.pyplot as plt
import jax.numpy as jnp

from ssm_jax.plotting import plot_inference, plot_uncertainty_ellipses
from ssm_jax.nlgssm.models import NonLinearGaussianSSM


def main():
    # Set up model for data generation and inference
    initial_mean = jnp.array([1.5, 0.0])
    state_dim = initial_mean.shape[0]
    nlgssm = NonLinearGaussianSSM(
        dynamics_function=lambda x: x + 0.4 * jnp.array([jnp.sin(x[1]), jnp.cos(x[0])]),
        dynamics_covariance=jnp.eye(state_dim) * 0.001,
        emission_function=lambda x: x,
        emission_covariance=jnp.eye(state_dim) * 0.05,
        initial_mean=initial_mean,
        initial_covariance=jnp.eye(state_dim),
    )

    # Sample data from model
    states, emissions = nlgssm.sample(key=0, num_timesteps=100)

    # Run EKF on emissions
    ekf_post = nlgssm.ekf_filter(emissions)
    ekf_means, ekf_covs = ekf_post.filtered_means, ekf_post.filtered_covariances

    # Plot true states and observations
    all_figures = {}
    fig, ax = plt.subplots()
    true_title = "Noisy obervations from hidden trajectory"
    _ = plot_inference(states, emissions, ax=ax, title=true_title)
    all_figures["ekf_spiral_true"] = fig

    # Plot EKF estimates
    fig, ax = plt.subplots()
    ekf_title = "EKF-filtered estimate of trajectory"
    ax = plot_inference(states, emissions, ekf_means, "EKF", ax=ax, title=ekf_title)
    # Add uncertainty ellipses to every fourth estimate
    plot_uncertainty_ellipses(ekf_means[::4], ekf_covs[::4], ax)
    all_figures["ekf_spiral_est"] = fig

    return all_figures


if __name__ == "__main__":
    figures = main()
    plt.show()
