"""Tests for the fitting methods shared by all state space models (``dynamax.ssm``)."""

import inspect
import os
import subprocess
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from jax import jit, vmap
from jax.tree_util import tree_leaves, tree_map, tree_structure

import dynamax.hidden_markov_model
from dynamax.hidden_markov_model import GaussianHMM, LogisticRegressionHMM
from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM, LinearGaussianSSM
from dynamax.utils.utils import ensure_array_has_batch_dim

NUM_TIMESTEPS = 50


def _gaussian_hmm():
    """An HMM with closed-form M-steps."""
    model = GaussianHMM(num_states=3, emission_dim=2, emission_prior_concentration=1.0, emission_prior_scale=1.0)
    params, props = model.initialize(jr.PRNGKey(0))
    _, emissions = model.sample(params, jr.PRNGKey(1), num_timesteps=NUM_TIMESTEPS)
    return model, params, props, emissions, None


def _logistic_regression_hmm():
    """An HMM with inputs and a gradient-based M-step."""
    model = LogisticRegressionHMM(num_states=3, input_dim=4, m_step_num_iters=5)
    inputs = jnp.ones((NUM_TIMESTEPS, 4))
    params, props = model.initialize(jr.PRNGKey(0))
    _, emissions = model.sample(params, jr.PRNGKey(1), num_timesteps=NUM_TIMESTEPS, inputs=inputs)
    return model, params, props, emissions, inputs


def _lgssm():
    """A linear Gaussian SSM with closed-form M-steps."""
    model = LinearGaussianSSM(state_dim=2, emission_dim=2)
    params, props = model.initialize(jr.PRNGKey(0))
    _, emissions = model.sample(params, jr.PRNGKey(1), num_timesteps=NUM_TIMESTEPS)
    return model, params, props, emissions, None


EXAMPLES = [_gaussian_hmm, _logistic_regression_hmm, _lgssm]


def _fit_em_python_loop(model, params, props, emissions, inputs, num_iters):
    """fit_em before it used lax.scan: a Python loop over a jitted EM step, kept as the reference."""
    batch_emissions = ensure_array_has_batch_dim(emissions, model.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, model.inputs_shape)

    @jit
    def em_step(params, m_step_state):
        batch_stats, lls = vmap(partial(model.e_step, params))(batch_emissions, batch_inputs)
        lp = model.log_prior(params) + lls.sum()
        params, m_step_state = model.m_step(params, props, batch_stats, m_step_state)
        return params, m_step_state, lp

    log_probs = []
    m_step_state = model.initialize_m_step_state(params, props)
    for _ in range(num_iters):
        params, m_step_state, marginal_logprob = em_step(params, m_step_state)
        log_probs.append(marginal_logprob)
    return params, jnp.array(log_probs)


def _assert_trees_allclose(actual, expected):
    """Assert that two parameter pytrees have the same structure and agree leaf by leaf."""
    assert tree_structure(actual) == tree_structure(expected)
    for a, b in zip(tree_leaves(actual), tree_leaves(expected)):
        assert jnp.allclose(a, b)


@pytest.mark.parametrize("make_example", EXAMPLES, ids=lambda f: f.__name__)
@pytest.mark.parametrize(["verbose", "print_every"], [(False, 1), (True, 1), (True, 2)],
                         ids=["silent", "every_iteration", "remainder_chunk"])
def test_fit_em_matches_python_loop(make_example, verbose, print_every):
    """The chunked-scan fit_em reproduces the original Python loop over EM steps."""
    model, params, props, emissions, inputs = make_example()
    loop_params, loop_lps = _fit_em_python_loop(model, params, props, emissions, inputs=inputs, num_iters=5)
    scan_params, scan_lps = model.fit_em(params, props, emissions, inputs=inputs, num_iters=5,
                                         verbose=verbose, print_every=print_every)
    assert scan_lps.shape == loop_lps.shape == (5,)
    assert jnp.allclose(scan_lps, loop_lps)
    _assert_trees_allclose(scan_params, loop_params)


@pytest.mark.parametrize("print_every", [0, 1.5])
def test_fit_em_rejects_invalid_print_every(print_every):
    """A bad print_every raises ValueError when verbose=True and is ignored when verbose=False."""
    model, params, props, emissions, _ = _gaussian_hmm()
    with pytest.raises(ValueError, match="print_every"):
        model.fit_em(params, props, emissions, num_iters=3, verbose=True, print_every=print_every)
    _, lps = model.fit_em(params, props, emissions, num_iters=3, verbose=False, print_every=print_every)
    assert lps.shape == (3,)


def test_fit_em_under_jit_vmap():
    """With verbose=False, fit_em can be vmapped over sequences and compiled as one program."""
    model, params, props, emissions, _ = _gaussian_hmm()
    batched = jnp.stack([emissions, emissions[::-1]])
    fit = jit(vmap(lambda y: model.fit_em(params, props, y, num_iters=3, verbose=False)))
    batched_params, batched_lps = fit(batched)
    assert batched_lps.shape == (2, 3)
    for i, y in enumerate(batched):
        ref_params, ref_lps = model.fit_em(params, props, y, num_iters=3, verbose=False)
        assert jnp.allclose(batched_lps[i], ref_lps)
        _assert_trees_allclose(tree_map(lambda x: x[i], batched_params), ref_params)


# ---------------------------------------------------------------------------------------------
# The EM scan carries (params, m_step_state), so initialize() must produce exactly the structure,
# shapes and dtypes the M-step returns. lax.scan checks that when fit_em is traced, so tracing
# fit_em (no compilation or execution) for every EM-capable model is the test.
# ---------------------------------------------------------------------------------------------

def _em_model_configs():
    """Every EM-capable model with default construction: (label, class, kwargs, integer emissions?)."""
    hmm = dynamax.hidden_markov_model
    return [
        ("BernoulliHMM", hmm.BernoulliHMM, dict(num_states=3, emission_dim=2), True),
        ("CategoricalHMM", hmm.CategoricalHMM, dict(num_states=3, emission_dim=2, num_classes=4), True),
        ("CategoricalRegressionHMM", hmm.CategoricalRegressionHMM, dict(num_states=3, num_classes=4, input_dim=2), True),
        ("GammaHMM", hmm.GammaHMM, dict(num_states=3), False),
        ("GaussianHMM", hmm.GaussianHMM, dict(num_states=3, emission_dim=2), False),
        ("DiagonalGaussianHMM", hmm.DiagonalGaussianHMM, dict(num_states=3, emission_dim=2), False),
        ("SphericalGaussianHMM", hmm.SphericalGaussianHMM, dict(num_states=3, emission_dim=2), False),
        ("SharedCovarianceGaussianHMM", hmm.SharedCovarianceGaussianHMM, dict(num_states=3, emission_dim=2), False),
        ("LowRankGaussianHMM", hmm.LowRankGaussianHMM, dict(num_states=3, emission_dim=2, emission_rank=1), False),
        ("GaussianMixtureHMM", hmm.GaussianMixtureHMM, dict(num_states=3, num_components=2, emission_dim=2), False),
        ("DiagonalGaussianMixtureHMM", hmm.DiagonalGaussianMixtureHMM, dict(num_states=3, num_components=2, emission_dim=2), False),
        ("LinearAutoregressiveHMM", hmm.LinearAutoregressiveHMM, dict(num_states=3, emission_dim=2, num_lags=1), False),
        ("LinearRegressionHMM", hmm.LinearRegressionHMM, dict(num_states=3, emission_dim=2, input_dim=2), False),
        ("LogisticRegressionHMM", hmm.LogisticRegressionHMM, dict(num_states=3, input_dim=2), True),
        ("MultinomialHMM", hmm.MultinomialHMM, dict(num_states=3, emission_dim=2, num_classes=4, num_trials=5), True),
        ("PoissonHMM", hmm.PoissonHMM, dict(num_states=3, emission_dim=2), True),
        ("LinearGaussianSSM", LinearGaussianSSM, dict(state_dim=2, emission_dim=2), False),
        ("LinearGaussianConjugateSSM", LinearGaussianConjugateSSM, dict(state_dim=2, emission_dim=2), False),
        ("LinearGaussianConjugateSSM-no-dynamics-bias", LinearGaussianConjugateSSM,
         dict(state_dim=2, emission_dim=2, has_dynamics_bias=False), False),
        ("LinearGaussianConjugateSSM-no-emissions-bias", LinearGaussianConjugateSSM,
         dict(state_dim=2, emission_dim=2, has_emissions_bias=False), False),
    ]


# Models whose initialize() accepts method="kmeans".
KMEANS_LABELS = ["GammaHMM", "GaussianHMM", "DiagonalGaussianHMM", "SphericalGaussianHMM",
                 "SharedCovarianceGaussianHMM", "LowRankGaussianHMM", "GaussianMixtureHMM",
                 "DiagonalGaussianMixtureHMM", "LinearAutoregressiveHMM", "LinearRegressionHMM",
                 "LogisticRegressionHMM"]
KMEANS_CONFIGS = [c for c in _em_model_configs() if c[0] in KMEANS_LABELS]


def _trace_fit_em(label, cls, kwargs, integer_emissions, method="prior"):
    """Build the model, initialize it with `method`, and trace fit_em."""
    model = cls(**kwargs)
    key = jr.PRNGKey(0)
    shape = (NUM_TIMESTEPS,) + model.emission_shape
    emissions = jr.randint(key, shape, 0, 2, dtype=jnp.int32) if integer_emissions else jr.normal(key, shape)
    inputs = None if model.inputs_shape is None else jr.normal(key, (NUM_TIMESTEPS,) + model.inputs_shape)
    init_kwargs = {} if method == "prior" else dict(method=method, emissions=emissions)
    if "inputs" in inspect.signature(model.initialize).parameters:  # only LogisticRegressionHMM
        init_kwargs["inputs"] = inputs
    params, props = model.initialize(key, **init_kwargs)
    jax.eval_shape(lambda: model.fit_em(params, props, emissions, inputs=inputs, num_iters=1, verbose=False))


@pytest.mark.parametrize(["label", "cls", "kwargs", "integer_emissions"], _em_model_configs(),
                         ids=[c[0] for c in _em_model_configs()])
def test_fit_em_carry_types_match_initialize(label, cls, kwargs, integer_emissions):
    """Prior-initialized parameters have the structure, shapes and dtypes the M-step returns."""
    _trace_fit_em(label, cls, kwargs, integer_emissions)


@pytest.mark.parametrize(["label", "cls", "kwargs", "integer_emissions"], KMEANS_CONFIGS,
                         ids=[c[0] for c in KMEANS_CONFIGS])
def test_fit_em_carry_types_match_kmeans_initialize(label, cls, kwargs, integer_emissions):
    """kmeans-initialized parameters also have the structure, shapes and dtypes the M-step returns."""
    _trace_fit_em(label, cls, kwargs, integer_emissions, method="kmeans")


def test_fit_em_carry_types_under_x64():
    """The two carry-type tests with 64-bit mode on, where priors built from Python floats used to sample float32."""
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-k", "carry_types_match"],
                            capture_output=True, text=True, env={**os.environ, "JAX_ENABLE_X64": "1"})
    assert result.returncode == 0, result.stdout[-2000:]
