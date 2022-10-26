import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, transforms
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


_COLOR_NAMES = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "clay",
    "pink",
    "greyish",
    "mint",
    "light cyan",
    "steel blue",
    "forest green",
    "pastel purple",
    "salmon",
    "dark brown",
]
COLORS = sns.xkcd_palette(_COLOR_NAMES)


def white_to_color_cmap(color, nsteps=256):
    """Return a cmap which ranges from white to the specified color.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    """
    # Get a red-white-black cmap
    cdict = {
        "red": ((0.0, 1.0, 1.0), (1.0, color[0], color[0])),
        "green": ((0.0, 1.0, 1.0), (1.0, color[1], color[0])),
        "blue": ((0.0, 1.0, 1.0), (1.0, color[2], color[0])),
    }
    cmap = LinearSegmentedColormap("white_color_colormap", cdict, nsteps)
    return cmap


def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1.0, 1.0))

    cdict = {"red": tuple(reds), "green": tuple(greens), "blue": tuple(blues), "alpha": tuple(alphas)}

    cmap = LinearSegmentedColormap("grad_colormap", cdict, nsteps)
    return cmap


CMAP = gradient_cmap(COLORS)

# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor="none", edgecolor="k", **kwargs):
    """Plot an ellipse to with centre `mu` and axes defined by `Sigma`."""
    cov = Sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # if facecolor not in kwargs:
    #     kwargs['facecolor'] = 'none'
    # if edgecolor not in kwargs:
    #     kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor, **kwargs
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_uncertainty_ellipses(means, Sigmas, ax, n_std=3.0, **kwargs):
    """Loop over means and Sigmas to add ellipses representing uncertainty."""
    for Sigma, mu in zip(Sigmas, means):
        plot_ellipse(Sigma, mu, ax, n_std, **kwargs)


def plot_lgssm_posterior(post_means, post_covs, ax=None, ellipse_kwargs={}, legend_kwargs={}, **kwargs):
    """Plot posterior means and covariances for the first two dimensions of
     the latent state of a LGSSM.

    Args:
        post_means: array(T, D).
        post_covs: array(T, D, D).
        ax: matplotlib axis.
        ellipse_kwargs: keyword arguments passed to matplotlib.patches.Ellipse().
        **kwargs: passed to ax.plot().
    """
    if ax is None:
        fig, ax = plt.subplots()

    # This is to stop some weird behaviour where running the function multiple
    # #  times with an empty argument wouldn't reset the dictionary.
    # if ellipse_kwargs is None:
    #     ellipse_kwargs = dict()

    # if 'edgecolor' not in ellipse_kwargs:
    #     if 'color' in kwargs:
    #         ellipse_kwargs['edgecolor'] = kwargs['color']

    # Select the first two dimensions of the latent space.
    post_means = post_means[:, :2]
    post_covs = post_covs[:, :2, :2]

    # Plot the mean trajectory
    ax.plot(post_means[:, 0], post_means[:, 1], **kwargs)
    # Plot covariance at each time point.
    plot_uncertainty_ellipses(post_means, post_covs, ax, **ellipse_kwargs)

    ax.axis("equal")

    if "label" in kwargs:
        ax.legend(**legend_kwargs)

    return ax


def plot_inference(states, emissions, estimates=None, est_type="", ax=None, title="", aspect=0.8, show_states=True):
    if ax is None:
        fig, ax = plt.subplots()
    if show_states:
        ax.plot(*states.T, label="True States")
    ax.plot(*emissions.T, "ok", fillstyle="none", ms=4, label="Observations")
    if estimates is not None:
        ax.plot(*estimates.T, color="r", linewidth=1.5, label=f"{est_type} Estimate")
    ax.set_aspect(aspect)
    ax.set_title(title)
    ax.legend(loc=1, borderpad=0.5, handlelength=4, fancybox=False, edgecolor="k")
    return ax
