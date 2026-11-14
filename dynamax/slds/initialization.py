"""
Data-driven initialization of an SLDS.

:func:`pca_initialization` fits the emission parameters by principal component analysis (PCA) and uses
them to estimate the continuous states. :func:`arhmm_initialization` then fits an autoregressive HMM
(ARHMM) to the estimated states, which gives the initial, transition and dynamics parameters.
:func:`pca_arhmm_initialization` combines the two and is what ``SLDS.initialize(method="pca_arhmm")`` calls.
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
    """Fit the emission parameters by PCA and estimate the continuous states from them.

    Any part of the emissions explained linearly by the inputs is removed first. The emission weights are
    the leading principal directions of what remains, the bias is its mean, and the noise variance is the
    variance left over in the other directions, shared across emission dimensions and kept above a small
    floor. With more than one subspace, each subspace gets a different random mix of the leading
    directions. The continuous states are estimated by inverting the emission model averaged over subspaces.

    Args:
        key: random number generator, used only when ``num_subspaces > 1``
        emissions: batch of emission sequences, shape (num_sequences, num_timesteps, emission_dim)
        state_dim: dimension of the continuous state; must not exceed ``emission_dim``
        inputs: optional batch of input sequences, shape (num_sequences, num_timesteps, input_dim)
        num_subspaces: number of emission subspaces
        cov_type: emission covariance type, ``"diag"`` or ``"full"``

    Returns:
        tuple of the emission parameters and the estimated continuous states, shape
        (num_sequences, num_timesteps, state_dim)
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


# ---- autoregressive HMM on x_hat -------------------------------------------------------------------
class _ParamsARHMM(NamedTuple):
    """Parameters of the autoregressive HMM used for initialization.

    :param initial: initial-state parameters of the SLDS
    :param transitions: transition parameters of the SLDS
    :param emissions: dynamics parameters of the SLDS, which act as the emissions of the HMM
    """
    initial: Any
    transitions: Any
    emissions: Any


class _ARHMMTransitions(HMMTransitions):
    """Wrap the SLDS transitions so the initialization HMM can use them.

    The HMM inputs are the estimated continuous states joined with the model inputs. Unlike other HMM
    transitions, this wrapper passes the whole input array to the SLDS transitions, which use the
    continuous state at each step and the inputs at the next step.
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
    """Wrap the SLDS dynamics so the initialization HMM can use them as emissions.

    The emissions are the estimated continuous states. Each state is predicted from the one before it,
    with zeros before the first step, as in :class:`LinearAutoregressiveHMM`.
    """

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
        """Previous continuous state at each time step, with zeros at the first step."""
        return jnp.concatenate([jnp.zeros((1, self.state_dim), dtype=x.dtype), x[:-1]])

    def _stats(self, params, smoothed_probs, x, inputs):
        return self.dynamics._regression_stats(smoothed_probs, self._lagged(x), x, inputs[:, self.state_dim:])

    def initialize(self, key=None, method="kmeans", emissions=None, inputs=None):
        """Initialize the dynamics parameters.

        Args:
            key: random number generator
            method: ``"prior"`` for a random start, or ``"kmeans"`` to cluster the estimated continuous
                states into ``num_states`` groups and fit each discrete state's dynamics to one group
            emissions: estimated continuous states, needed for ``"kmeans"``
            inputs: inputs of the HMM, needed for ``"kmeans"``

        Returns:
            tuple of parameters and their properties
        """
        params, props = self.dynamics.initialize(key, "prior")
        if method == "kmeans":
            if emissions is None or inputs is None:
                raise ValueError("method='kmeans' needs the emissions (x_hat) and inputs.")
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
    """Autoregressive HMM built from the SLDS initial, transition and dynamics components."""

    def __init__(self, model: SLDS):
        super().__init__(model.num_states, model.initial_component,
                         _ARHMMTransitions(model.transition_component, model.state_dim),
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
    """Fit the initial, transition and dynamics parameters with an autoregressive HMM.

    The HMM is fitted by EM to the estimated continuous states. Each restart begins from a uniform
    initial distribution and the transition component's ``"sticky"`` initialization, which favors staying
    in the same state except for recurrent-only transitions, whose logits and recurrent weights start at
    zero. The restart with the highest log joint probability (log prior plus log likelihood of the
    estimated states) is kept.

    Args:
        key: random number generator
        model: the SLDS being initialized
        x_hat: estimated continuous states, shape (num_sequences, num_timesteps, state_dim)
        inputs: batch of input sequences, shape (num_sequences, num_timesteps, input_dim)
        num_iters: number of iterations of EM per restart
        num_restarts: number of restarts
        init_method: how each restart sets the starting dynamics, ``"kmeans"`` or ``"prior"``

    Returns:
        tuple of the initial, transition and dynamics parameters, and the log joint probability of the
        best restart
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
    """Initialize the SLDS parameters from data with PCA and an autoregressive HMM.

    The emission parameters come from :func:`pca_initialization`, and the initial, transition and
    dynamics parameters from :func:`arhmm_initialization` on the estimated continuous states.

    Args:
        key: random number generator
        model: the SLDS being initialized
        params: model parameters (not used)
        props: properties of the parameters (not used)
        emissions: one sequence or a batch of sequences of emissions
        inputs: optional inputs matching ``emissions``
        num_init_iters: number of iterations of EM for the autoregressive HMM
        num_init_restarts: number of restarts of the autoregressive HMM
        init_method: how the autoregressive HMM sets its starting dynamics, ``"kmeans"`` or ``"prior"``

    Returns:
        initialized model parameters
    """
    batch_emissions, batch_inputs = model._batch(emissions, inputs)
    emission_component = model.emission_component
    k_pca, k_arhmm = jr.split(key)
    emission_params, x_hat = pca_initialization(k_pca, batch_emissions, model.state_dim, batch_inputs,
                                                emission_component.num_subspaces, emission_component.cov_type)
    initial, transitions, dynamics, _ = arhmm_initialization(k_arhmm, model, x_hat, batch_inputs,
                                                             num_init_iters, num_init_restarts, init_method)
    return ParamsSLDS(initial=initial, transitions=transitions, dynamics=dynamics, emissions=emission_params)
