r"""
Data-driven initialization of an SLDS, as in ``lindermanlab/ssm``.

:func:`pca_initialization` fits the emission model by principal component analysis and inverts it to a
point estimate :math:`\hat x` of the continuous states. :func:`arhmm_initialization` then fits an
autoregressive HMM to :math:`\hat x` by EM, which gives the initial, transition and dynamics
parameters. :func:`pca_arhmm_initialization` composes the two and is what
``SLDS.initialize(method="pca_arhmm")`` calls.

Differences from ``ssm`` (deliberate): the ARHMM starts from a k-means clustering of :math:`\hat x`
instead of random one-hot assignments; its first time step uses dynamax's ARHMM convention (a
zero-padded lag, :meth:`LinearAutoregressiveHMM.compute_inputs`) rather than ``ssm``'s separate initial
density; with one emission subspace the PCA basis is not randomly rotated; the PCA noise variance is
clamped at ``1e-6`` (``ssm`` takes the log of zero when ``emission_dim == state_dim``).
"""
from typing import Any, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMTransitions
from dynamax.slds.abstractions import ParamsSLDS, SLDS, SLDSDynamics, SLDSTransitions
from dynamax.slds.emissions import ParamsSLDSEmissions
from dynamax.types import PRNGKeyT, Scalar
from dynamax.utils.cluster import kmeans

MIN_NOISE_VARIANCE = 1e-6


def pca_initialization(key: PRNGKeyT,
                       emissions: Float[Array, "num_sequences num_timesteps emission_dim"],
                       state_dim: int,
                       inputs: Optional[Float[Array, "num_sequences num_timesteps input_dim"]] = None,
                       num_subspaces: int = 1,
                       cov_type: str = "diag"
                       ) -> Tuple[ParamsSLDSEmissions, Float[Array, "num_sequences num_timesteps state_dim"]]:
    r"""Emission parameters from PCA and the inverted continuous states (``ssm``'s ``_initialize_with_pca``).

    With inputs, :math:`D` is first fitted by least squares of the stacked emissions on the stacked inputs
    (no intercept) and removed. The residuals are centered (:math:`d` = their mean) and decomposed by SVD.
    With one subspace :math:`H` is the top-``state_dim`` principal directions. With ``num_subspaces > 1``
    the top ``min(state_dim * num_subspaces, emission_dim)`` directions are mixed by a random row-orthonormal
    matrix per subspace, so the subspaces differ. The noise variance is the mean of the remaining
    eigenvalues (as ``sklearn``), at least ``MIN_NOISE_VARIANCE``, shared across dimensions:
    :math:`R = \sigma^2 I`. Finally :math:`\hat x_t = \bar H^+ (y_t - \bar d - \bar D u_t)` with the
    subspace-averaged parameters.

    Returns:
        the emission parameters and :math:`\hat x` with shape (num_sequences, num_timesteps, state_dim).
    """
    num_sequences, num_timesteps, emission_dim = emissions.shape
    if emission_dim < state_dim:
        raise ValueError(f"PCA initialization needs emission_dim >= state_dim, got {emission_dim} < {state_dim}.")
    input_dim = 0 if inputs is None else inputs.shape[-1]
    y = emissions.reshape(-1, emission_dim)
    if input_dim > 0:
        u = inputs.reshape(-1, input_dim)
        input_weights = jnp.linalg.lstsq(u, y)[0].T                      # (N, M)
        y = y - u @ input_weights.T
    else:
        input_weights = jnp.zeros((emission_dim, 0))
    bias = y.mean(0)
    centered = y - bias
    num_components = min(state_dim * num_subspaces, emission_dim)
    if centered.shape[0] < num_components:
        raise ValueError(f"PCA initialization needs at least {num_components} time points in total, got {centered.shape[0]}.")
    _, singular_values, vt = jnp.linalg.svd(centered, full_matrices=False)
    components = vt[:num_components]                                     # (C, N), orthonormal rows
    eigenvalues = singular_values ** 2 / max(centered.shape[0] - 1, 1)
    trailing = eigenvalues[num_components:]
    noise_variance = trailing.mean() if trailing.size > 0 else jnp.asarray(0.0)
    noise_variance = jnp.maximum(noise_variance, MIN_NOISE_VARIANCE)

    if num_subspaces == 1:
        weights = components.T[None]                                     # (1, N, D)
    else:
        def _mix(k):
            omega = jnp.linalg.svd(jr.normal(k, (state_dim, num_components)), full_matrices=False)[2]   # (D, C) orthonormal rows
            return (omega @ components).T                                                               # (N, D)
        weights = vmap(_mix)(jr.split(key, num_subspaces))
    if cov_type == "diag":
        covs = noise_variance * jnp.ones((num_subspaces, emission_dim))
    elif cov_type == "full":
        covs = noise_variance * jnp.tile(jnp.eye(emission_dim), (num_subspaces, 1, 1))
    else:
        raise ValueError(f"cov_type must be 'full' or 'diag', got {cov_type!r}")
    params = ParamsSLDSEmissions(weights=weights,
                                 biases=jnp.tile(bias, (num_subspaces, 1)),
                                 input_weights=jnp.tile(input_weights, (num_subspaces, 1, 1)),
                                 covs=covs)

    # Invert the subspace-averaged emission model (the same formula as LinearGaussianSLDSEmissions.invert).
    H_bar, D_bar, d_bar = weights.mean(0), params.input_weights.mean(0), params.biases.mean(0)
    full_inputs = jnp.zeros((num_sequences, num_timesteps, 0)) if inputs is None else inputs
    residuals = emissions - full_inputs @ D_bar.T - d_bar
    x_hat = residuals @ jnp.linalg.pinv(H_bar).T
    return params, x_hat


# ---- autoregressive HMM on x̂ ----------------------------------------------------------------------
class _ParamsARHMM(NamedTuple):
    """Parameters of the initialization ARHMM: the SLDS initial, transition and dynamics parameters."""
    initial: Any
    transitions: Any
    emissions: Any


class _ARHMMTransitions(HMMTransitions):
    r"""``HMMTransitions`` adapter around an :class:`SLDSTransitions`.

    The ARHMM inputs are :math:`[\hat x_t, u_t]`; the SLDS component reads :math:`\hat x_t` and
    :math:`u_{t+1}` for the pair :math:`t \to t+1`, so this adapter passes the whole input array
    through instead of ``HMMTransitions``' ``inputs[1:]`` convention.
    """

    def __init__(self, transitions: SLDSTransitions, state_dim: int):
        self.transitions = transitions
        self.state_dim = state_dim
        self.num_states = transitions.num_states

    def distribution(self, params, state, inputs=None):
        raise NotImplementedError("The initialization ARHMM is not sampled.")

    def initialize(self, key=None, method="prior", **kwargs):
        return self.transitions.initialize(key, method, **kwargs)

    def log_prior(self, params) -> Scalar:
        return self.transitions.log_prior(params)

    def _compute_transition_matrices(self, params, inputs=None):
        D = self.state_dim
        return self.transitions.transition_matrices(params, inputs[:, :D], inputs[:, D:])

    def collect_suff_stats(self, params, posterior, inputs=None):
        D = self.state_dim
        return self.transitions.collect_suff_stats(params, posterior, inputs[:, :D][None], inputs[:, D:])

    def initialize_m_step_state(self, params, props):
        return self.transitions.initialize_m_step_state(params, props)

    def m_step(self, params, props, batch_stats, m_step_state):
        return self.transitions.m_step(params, props, batch_stats, m_step_state)


class _ARHMMEmissions(HMMEmissions):
    r"""``HMMEmissions`` adapter around an :class:`SLDSDynamics`: the "emissions" are :math:`\hat x_t`
    and the lag is the zero-padded :math:`\hat x_{t-1}`, as :class:`LinearAutoregressiveHMM`."""

    def __init__(self, dynamics: SLDSDynamics):
        self.dynamics = dynamics
        self.num_states = dynamics.num_states
        self.state_dim = dynamics.state_dim

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.state_dim,)

    def distribution(self, params, state, inputs=None):
        raise NotImplementedError("The initialization ARHMM is not sampled.")

    def log_prior(self, params) -> Scalar:
        return self.dynamics.log_prior(params)

    def _lagged(self, x: Array) -> Array:
        """x_{t-1} for every t, with x_{-1} = 0 (``LinearAutoregressiveHMM.compute_inputs`` with one lag)."""
        return jnp.concatenate([jnp.zeros((1, self.state_dim), dtype=x.dtype), x[:-1]])

    def _stats(self, params, smoothed_probs, x, inputs):
        return self.dynamics.collect_suff_stats_from_pairs(params, smoothed_probs, self._lagged(x), x, inputs[:, self.state_dim:])

    def initialize(self, key=None, method="kmeans", emissions=None, inputs=None):
        """``"prior"``: the dynamics component's random start. ``"kmeans"``: cluster all pooled
        :math:`\\hat x_t` into ``num_states`` groups and take one closed-form M-step from the one-hot assignments."""
        params, props = self.dynamics.initialize(key, "prior")
        if method == "kmeans":
            if emissions is None or inputs is None:
                raise ValueError("method='kmeans' needs the emissions (x̂) and inputs.")
            num_sequences, num_timesteps, _ = emissions.shape
            assignments = kmeans(emissions.reshape(-1, self.state_dim), self.num_states, key).assignments
            smoothed_probs = jnp.eye(self.num_states)[assignments].reshape(num_sequences, num_timesteps, self.num_states)
            stats = vmap(self._stats, (None, 0, 0, 0))(params, smoothed_probs, emissions, inputs)
            params, _ = self.dynamics.m_step(params, props, stats, None)
        elif method != "prior":
            raise ValueError(f"Unknown initialization method {method!r}")
        return params, props

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        return self.dynamics.ar_log_likelihoods(params, self._lagged(emissions), emissions, inputs[:, self.state_dim:])

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        return self._stats(params, posterior.smoothed_probs, emissions, inputs)

    def initialize_m_step_state(self, params, props):
        return self.dynamics.initialize_m_step_state(params, props)

    def m_step(self, params, props, batch_stats, m_step_state):
        return self.dynamics.m_step(params, props, batch_stats, m_step_state)


class _ARHMM(HMM):
    """The initialization ARHMM: the SLDS initial and transition components with the dynamics as emissions."""

    def __init__(self, model: SLDS):
        super().__init__(model.num_states, model.initial_component,
                         model.transition_component.as_hmm_transitions(model.state_dim),
                         _ARHMMEmissions(model.dynamics_component))
        self.state_dim = model.state_dim
        self.input_dim = model.input_dim

    @property
    def inputs_shape(self) -> Tuple[int]:
        return (self.state_dim + self.input_dim,)

    def initialize(self, key, method="kmeans", emissions=None, inputs=None, initial_params=None, transition_params=None
                   ) -> Tuple[_ParamsARHMM, _ParamsARHMM]:
        k1, k2, k3 = jr.split(key, 3)
        p_init, pr_init = self.initial_component.initialize(k1, "prior", **(initial_params or {}))
        p_trans, pr_trans = self.transition_component.initialize(k2, **(transition_params or {}))
        p_emis, pr_emis = self.emission_component.initialize(k3, method, emissions=emissions, inputs=inputs)
        return _ParamsARHMM(p_init, p_trans, p_emis), _ParamsARHMM(pr_init, pr_trans, pr_emis)


def arhmm_initialization(key: PRNGKeyT,
                         model: SLDS,
                         x_hat: Float[Array, "num_sequences num_timesteps state_dim"],
                         inputs: Float[Array, "num_sequences num_timesteps input_dim"],
                         num_iters: int = 50,
                         num_restarts: int = 1,
                         init_method: str = "kmeans") -> Tuple[Any, Any, Any, Scalar]:
    r"""Initial, transition and dynamics parameters from an ARHMM fitted to :math:`\hat x` by EM.

    As in ``ssm``, the ARHMM starts from a uniform initial distribution and the transition component's
    ``"sticky"`` start (:math:`0.95 I + 0.05 U` row-normalized for a transition matrix). Each of the
    ``num_restarts`` restarts uses ``jax.random.fold_in(key, r)``; the restart with the highest
    log joint probability (log prior plus marginal log likelihood of :math:`\hat x`) is returned,
    together with that value.
    """
    arhmm = _ARHMM(model)
    arhmm_inputs = jnp.concatenate([x_hat, inputs], axis=-1)
    uniform = dict(initial_probs=jnp.ones(model.num_states) / model.num_states)
    best = None
    for r in range(num_restarts):
        params, props = arhmm.initialize(jr.fold_in(key, r), init_method, x_hat, arhmm_inputs,
                                         initial_params=uniform, transition_params=dict(method="sticky"))
        params, _ = arhmm.fit_em(params, props, x_hat, arhmm_inputs, num_iters=num_iters, verbose=False)
        score = arhmm.log_prior(params) + vmap(arhmm.marginal_log_prob, (None, 0, 0))(params, x_hat, arhmm_inputs).sum()
        if best is None or score > best[1]:
            best = (params, score)
    params, score = best
    return params.initial, params.transitions, params.emissions, score


def pca_arhmm_initialization(key: PRNGKeyT,
                             model: SLDS,
                             params: ParamsSLDS,
                             props: ParamsSLDS,
                             emissions: Array,
                             inputs: Optional[Array] = None,
                             num_init_iters: int = 50,
                             num_init_restarts: int = 1,
                             init_method: str = "kmeans") -> ParamsSLDS:
    """``ssm``'s SLDS initialization: emissions from :func:`pca_initialization`, the rest from
    :func:`arhmm_initialization` on the inverted states. ``emissions`` may be one sequence or a batch."""
    batch_emissions, batch_inputs = model._batch(emissions, inputs)
    emission_component = model.emission_component
    k_pca, k_arhmm = jr.split(key)
    emission_params, x_hat = pca_initialization(k_pca, batch_emissions, model.state_dim, batch_inputs,
                                                emission_component.num_subspaces, emission_component.cov_type)
    initial, transitions, dynamics, _ = arhmm_initialization(k_arhmm, model, x_hat, batch_inputs,
                                                             num_init_iters, num_init_restarts, init_method)
    return ParamsSLDS(initial=initial, transitions=transitions, dynamics=dynamics, emissions=emission_params)
