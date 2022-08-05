import matplotlib.pyplot as plt
import jax.numpy as jnp

from ssm_jax.plotting import plot_inference, plot_uncertainty_ellipses
from ssm_jax.nlgssm.models import NonLinearGaussianSSM
from ssm_jax.ukf.inference import UKFHyperParams


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
    hyperparams = UKFHyperParams(alpha=10, beta=10, kappa=10)
    ukf_post = nlgssm.ukf_filter(emissions, hyperparams)
    ukf_means, ukf_covs = ukf_post.filtered_means, ukf_post.filtered_covariances

    # Plot true states and observations
    all_figures = {}
    fig, ax = plt.subplots()
    true_title = "Noisy obervations from hidden trajectory"
    _ = plot_inference(states, emissions, ax=ax, title=true_title)
    all_figures["ukf_spiral_true"] = fig

    # Plot UKF estimates
    fig, ax = plt.subplots()
    ekf_title = "UKF-filtered estimate of trajectory"
    ax = plot_inference(states, emissions, ukf_means, "UKF", ax=ax, title=ekf_title)
    # Add uncertainty ellipses to every fourth estimate
    plot_uncertainty_ellipses(ukf_means[::4], ukf_covs[::4], ax)
    all_figures["ukf_spiral_est"] = fig

    return all_figures


if __name__ == "__main__":
    figures = main()
    plt.show()
