"""
Recovery against ``lindermanlab/ssm`` (commit eb6c8aa) on the two notebook problems: on ssm's own datasets, dynamax's
fit and ssm's saved fit (``fixtures/ssm_fixtures.npz``) are scored on discrete state, continuous state and parameter
recovery, and each dynamax score must be at least ssm's minus 1% of its magnitude. The fits run in 64-bit
floats, as ssm does.
"""
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from dynamax.slds import GaussianSLDS, RecurrentGaussianSLDS
from dynamax.utils.utils import find_permutation

X64 = jax.config.read("jax_enable_x64")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ssm_fixtures.npz")
MODELS = {"slds": lambda: GaussianSLDS(5, 2, 10),
          "rslds": lambda: RecurrentGaussianSLDS(4, 2, 10, recurrent_only=True, dynamics_cov_type="diag")}
GROUPS = {"slds": ["P", "A", "b", "Q", "C", "d", "R"],
          "rslds": ["logits", "recurrent_weights", "A", "b", "Q", "C", "d", "R"]}


def natural_params(params):
    """dynamax parameters in natural form, keyed as the fixture's ``recovery/<model>/true/`` and ``ssm/`` entries."""
    Q = params.dynamics.covs
    out = {"A": params.dynamics.weights, "b": params.dynamics.biases, "Q": Q if Q.ndim == 3 else jax.vmap(jnp.diag)(Q),
           "C": params.emissions.weights[0], "d": params.emissions.biases[0], "R": params.emissions.covs[0]}
    if hasattr(params.transitions, "transition_matrix"):
        out["P"] = params.transitions.transition_matrix
    else:
        out["logits"], out["recurrent_weights"] = params.transitions.logits[0], params.transitions.recurrent_weights
    return {k: np.asarray(v) for k, v in out.items()}


def in_true_coordinates(p, perm, coef):
    """Fitted parameters relabeled (fitted state k is true state perm[k]) and mapped by x = G x̂ + g, the
    least-squares fit of x on [x̂, 1] with coefficients ``coef``."""
    inv = np.argsort(perm)                                  # true state j came from fitted state inv[j]
    G, g = coef[:-1].T, coef[-1]
    G_inv = np.linalg.inv(G)
    A = G @ p["A"][inv] @ G_inv
    C = p["C"] @ G_inv
    out = {"A": A, "b": p["b"][inv] @ G.T + g - A @ g, "Q": G @ p["Q"][inv] @ G.T, "C": C, "d": p["d"] - C @ g,
           "R": p["R"]}
    if "P" in p:
        out["P"] = p["P"][inv][:, inv]
    else:
        W = p["recurrent_weights"][inv] @ G_inv
        out["recurrent_weights"], out["logits"] = W, p["logits"][inv] - W @ g
    return out


def variance_explained(fitted, true, group):
    if group in ("logits", "recurrent_weights"):            # softmax logits: remove the common shift over states
        fitted, true = fitted - fitted.mean(0), true - true.mean(0)
    return 1.0 - np.sum((fitted - true) ** 2) / np.sum(true ** 2)


def recovery(name, p, x_hat, z_hat, data):
    """Discrete state recovery, continuous state recovery and parameter recovery (per group) of one fit."""
    x, z = data[f"recovery/{name}/x"], data[f"recovery/{name}/z"]
    perm = np.asarray(find_permutation(z_hat, z))
    discrete = np.mean(perm[z_hat] == z)
    design = np.column_stack([x_hat, np.ones(len(x_hat))])
    coef = np.linalg.lstsq(design, x, rcond=None)[0]
    continuous = 1.0 - np.sum((design @ coef - x) ** 2) / np.sum((x - x.mean(0)) ** 2)
    aligned = in_true_coordinates(p, perm, coef)
    parameters = {k: variance_explained(aligned[k], data[f"recovery/{name}/true/{k}"], k) for k in GROUPS[name]}
    return {"discrete state recovery": float(discrete), "continuous state recovery": float(continuous), **parameters}


@pytest.mark.parametrize("name", list(MODELS))
def test_recovery_is_at_least_as_good_as_ssm(name):
    data = np.load(FIXTURE)
    y = jnp.asarray(data[f"recovery/{name}/y"])
    model = MODELS[name]()
    params, props = model.initialize(jr.PRNGKey(0), method="pca_arhmm", emissions=y)
    fitted, _, posterior = model.fit_em(params, props, y, verbose=False, return_posterior=True)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)   # one sequence: drop the batch axis
    x_hat = np.asarray(posterior.continuous.smoothed_means)
    z_hat = np.asarray(model.most_likely_states(fitted, y, posterior))
    ours = recovery(name, natural_params(fitted), x_hat, z_hat, data)
    theirs = recovery(name, {k: data[f"recovery/{name}/ssm/{k}"] for k in GROUPS[name]}, data[f"recovery/{name}/x_hat"],
                      data[f"recovery/{name}/z_hat"], data)
    assert np.isclose(theirs["discrete state recovery"], data[f"recovery/{name}/discrete_state_recovery"])
    print(f"\n{name}")
    for k in ours:
        print(f"  {k:>26s}: dynamax {ours[k]:.4f} vs ssm {theirs[k]:.4f}")
    for k in ours:
        assert ours[k] >= theirs[k] - 0.01 * abs(theirs[k]), k
test_recovery_is_at_least_as_good_as_ssm.__test__ = X64


def test_recovery_under_x64():
    """Run the recovery test in a 64-bit subprocess (collected only under float32)."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-4000:]
test_recovery_under_x64.__test__ = not X64
