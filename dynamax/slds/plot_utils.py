import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from matplotlib.patches import Ellipse, transforms
from mpl_toolkits.mplot3d import Axes3D


# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor='none', edgecolor='k', plot_center='true', **kwargs):
    cov = Sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (transforms.Affine2D()
                        .rotate_deg(45)
                        .scale(scale_x, scale_y)
                        .translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)

    if plot_center:
        ax.plot(mean_x, mean_y, '.')
    return ax.add_patch(ellipse)


def savedotfile(dotfiles):
    if "FIGDIR" in os.environ:
        figdir = os.environ["FIGDIR"]
        for name, dot in dotfiles.items():
            fname_full = os.path.join(figdir, name)
            dot.render(fname_full)
            print(f"saving dot file to {fname_full}")


def savefig(figures, *args, **kwargs):
    if "FIGDIR" in os.environ:
        figdir = os.environ["FIGDIR"]
        for name, figure in figures.items():
            fname_full = os.path.join(figdir, name)
            print(f"saving image to {fname_full}")
            figure.savefig(f"{fname_full}.pdf", *args, **kwargs)
            figure.savefig(f"{fname_full}.png", *args, **kwargs)


def scale_3d(ax, x_scale, y_scale, z_scale, factor):    
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])    
    scale=scale*(1.0/scale.max())    
    scale[3,3]=factor
    def short_proj():    
        return np.dot(Axes3D.get_proj(ax), scale)    
    return short_proj    

def style3d(ax, x_scale, y_scale, z_scale, factor=0.62):
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.get_proj = scale_3d(ax, x_scale, y_scale, z_scale, factor)


def kdeg(x, X, h):
    """
    KDE under a gaussian kernel

    Parameters
    ----------
    x: array(eval, D)
    X: array(obs, D)
    h: float

    Returns
    -------
    array(eval):
        KDE around the observed values
    """
    N, D = X.shape
    nden, _ = x.shape

    Xhat = X.reshape(D, 1, N)
    xhat = x.reshape(D, nden, 1)
    u = xhat - Xhat
    u = linalg.norm(u, ord=2, axis=0) ** 2 / (2 * h ** 2)
    px = np.exp(-u).sum(axis=1) / (N * h * np.sqrt(2 * np.pi))
    return px
