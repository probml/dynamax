"""
Recovery benchmark against ``lindermanlab/ssm``: on the data of ``fixtures/ssm_benchmark.npz`` (the tutorial
problem, one sequence per seed, with ssm's final state accuracy and aligned latent RMSE), dynamax's Laplace EM
from the PCA + ARHMM start must match ssm in median (accuracy within 0.02, RMSE within 10%). The fixture is
exported by ``fixtures/export_ssm_fixtures.py --benchmark`` (see ``fixtures/README.md``).
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.slds import GaussianSLDS
from dynamax.utils.utils import find_permutation

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ssm_benchmark.npz")


def fit_and_score(y, z, x, key, num_iters=50):
    model = GaussianSLDS(3, 2, y.shape[1])
    params, props = model.initialize(key, method="pca_arhmm", emissions=y)
    fitted, _ = model.fit_em(params, props, y, num_iters=num_iters, verbose=False)
    posterior, _ = model.approximate_posterior(fitted, y, num_iters=5)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
    z_hat = model.most_likely_states(fitted, y, posterior)
    accuracy = jnp.mean(find_permutation(z_hat, z)[z_hat] == z)
    design = jnp.concatenate([posterior.continuous.smoothed_means, jnp.ones((y.shape[0], 1))], 1)
    rmse = jnp.sqrt(jnp.mean((design @ jnp.linalg.lstsq(design, x)[0] - x) ** 2))
    return float(accuracy), float(rmse)


def test_recovery_is_at_least_as_good_as_ssm():
    data = np.load(FIXTURE)
    num_seeds = int(data["num_seeds"])
    ours, theirs = [], []
    for seed in range(num_seeds):
        y, z, x = (jnp.asarray(data[f"seed{seed}/{k}"]) for k in ["y", "z", "x"])
        ours.append(fit_and_score(y, z, x, jr.PRNGKey(seed)))
        theirs.append((float(data[f"seed{seed}/accuracy"]), float(data[f"seed{seed}/rmse"])))
    ours, theirs = np.array(ours), np.array(theirs)
    print(f"\naccuracy: dynamax median {np.median(ours[:, 0]):.3f} vs ssm {np.median(theirs[:, 0]):.3f}; "
          f"rmse: dynamax median {np.median(ours[:, 1]):.3f} vs ssm {np.median(theirs[:, 1]):.3f}")
    assert np.median(ours[:, 0]) >= np.median(theirs[:, 0]) - 0.02
    assert np.median(ours[:, 1]) <= np.median(theirs[:, 1]) * 1.1 + 1e-3
