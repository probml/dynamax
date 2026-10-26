r"""
Discrete transition components of a switching linear dynamical system.

:class:`StationarySLDSTransitions` is a fixed transition matrix (the plain SLDS).
:class:`RecurrentSLDSTransitions` (rSLDS) makes the next state depend on the continuous state,
:math:`P_t[i, :] = \mathrm{softmax}(\mathrm{logits}[i, :] + W_x x_t + W_u u_{t+1})`.
"""
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import optax.tree_utils as otu
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax, vmap
from jax.scipy.special import xlogy
from jax.tree_util import tree_map
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties
from dynamax.slds.abstractions import DiscretePosteriorSLDS, RecurrentTransStats, SLDSTransitions
from dynamax.types import PRNGKeyT, Scalar

ParamsStationarySLDSTransitions = ParamsStandardHMMTransitions


class StationarySLDSTransitions(SLDSTransitions):
    r"""A fixed transition matrix :math:`P` with a Dirichlet prior on its rows,
    :math:`P_k \sim \mathrm{Dir}(\beta 1_K + \kappa e_k)` (``ssm``: ``StationaryTransitions`` at
    :math:`\beta = 1, \kappa = 0`, ``StickyTransitions`` with :math:`\kappa > 0`).

    :param num_states: :math:`K`
    :param concentration: :math:`\beta`; 1.0 (the default) is the maximum likelihood update
    :param stickiness: :math:`\kappa`
    """
    depends_on_x = False

    def __init__(self, num_states: int, concentration: Union[Scalar, Array] = 1.0, stickiness: Union[Scalar, Array] = 0.0):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)

    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior",
                   transition_matrix: Optional[Array] = None) -> Tuple[ParamsStationarySLDSTransitions, ParamsStationarySLDSTransitions]:
        """A Dirichlet draw per row (``method="prior"``), ``ssm``'s sticky random start
        :math:`0.95 I + 0.05 U` row-normalized (``method="sticky"``, used by the ARHMM initialization),
        or a user-supplied matrix; and the properties."""
        if transition_matrix is None:
            if key is None:
                raise ValueError("key must be provided if transition_matrix is not provided.")
            if method == "prior":
                transition_matrix = jr.dirichlet(key, self.concentration)
            elif method == "sticky":
                K = self.num_states
                transition_matrix = 0.95 * jnp.eye(K) + 0.05 * jr.uniform(key, (K, K))
                transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"Unknown initialization method {method!r}")
        params = ParamsStationarySLDSTransitions(transition_matrix=transition_matrix)
        props = ParamsStationarySLDSTransitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def log_prior(self, params: ParamsStationarySLDSTransitions) -> Scalar:
        """Dirichlet log prior of the rows; zero for a single state."""
        if self.num_states == 1:
            return 0.0
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def transition_matrices(self, params, x, inputs) -> Array:
        """The (K, K) matrix."""
        return params.transition_matrix

    def log_transition_matrices(self, params, x, inputs) -> Array:
        """log of the (K, K) matrix."""
        return jnp.log(params.transition_matrix)

    def transition_row(self, params, state, x, next_input) -> Array:
        """Row ``state`` of the matrix."""
        return params.transition_matrix[state]

    def expected_log_trans_prob(self, params, x, inputs, trans_probs) -> Scalar:
        """sum_{ij} Ezzp1[i, j] log P[i, j], with the pairwise expectations summed over time."""
        return jnp.sum(xlogy(trans_probs, params.transition_matrix))

    def neg_hessian_x(self, params, x, inputs, trans_probs) -> Array:
        """Zero: the term does not depend on x."""
        num_timesteps, state_dim = x.shape
        return jnp.zeros((num_timesteps - 1, state_dim, state_dim))

    def collect_suff_stats(self, params, qz: DiscretePosteriorSLDS, x_samples, inputs) -> Array:
        """The (K, K) expected transition counts."""
        return qz.trans_probs

    def initialize_m_step_state(self, params, props) -> Any:
        return None

    def m_step(self, params, props, batch_stats, m_step_state) -> Tuple[ParamsStationarySLDSTransitions, Any]:
        r"""Floored normalized counts, :math:`P_{ij} \propto \max(\sum_b \mathbb{E}[n_{ij}] + \beta_{ij} - 1, 0) + 10^{-32} n_i`
        with :math:`n_i` the row total (``ssm``'s ``StationaryTransitions.m_step`` with its ``1e-32`` floor made relative,
        so it is not flushed to zero in float32; the Dirichlet mode where it is defined)."""
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.ones((1, 1))
            else:
                counts = jnp.maximum(batch_stats.sum(axis=0) + (self.concentration - 1.0), 0.0)
                counts = counts + 1e-32 * jnp.maximum(counts.sum(axis=-1, keepdims=True), 1.0)
                transition_matrix = counts / counts.sum(axis=-1, keepdims=True)
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state

    def as_hmm_transitions(self, state_dim: int):
        """The ``HMMTransitions`` adapter used by the ARHMM initialization."""
        from dynamax.slds.initialization import _ARHMMTransitions
        return _ARHMMTransitions(self, state_dim)


class ParamsRecurrentSLDSTransitions(NamedTuple):
    r"""Parameters of recurrent transitions (``ssm``: ``log_Ps``/``r``, ``Rs``, ``Ws``).

    .. math::
        \log p(z_{t+1} = j \mid z_t = i, x_t, u_{t+1}) = \mathrm{log\,softmax}_j(\mathrm{logits}[i, j] + W_x[j] \cdot x_t + W_u[j] \cdot u_{t+1})

    :param logits: (K, K), or (1, K) with ``recurrent_only`` (``ssm``'s bias ``r``)
    :param recurrent_weights: :math:`W_x`, (K, D)
    :param input_weights: :math:`W_u`, (K, M)
    """
    logits: Union[Float[Array, "num_states num_states"], Float[Array, "1 num_states"], ParameterProperties]
    recurrent_weights: Union[Float[Array, "num_states state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]


class RecurrentSLDSTransitions(SLDSTransitions):
    r"""Transitions whose next-state probabilities depend on the continuous state (the rSLDS of Linderman
    et al., 2017; ``ssm``'s ``RecurrentTransitions`` and, with ``recurrent_only``, ``RecurrentOnlyTransitions``).

    The log prior is a Dirichlet on the rows of :math:`\mathrm{softmax}(\mathrm{logits})` (not with
    ``recurrent_only``) plus :math:`-\tfrac12 \lambda_x \|W_x\|^2 - \tfrac12 \lambda_u \|W_u\|^2`. The M-step
    maximizes the expected log joint numerically, warm-started at the current parameters, with ``optax.lbfgs``
    by default, stopped as ``scipy``'s L-BFGS-B stops it in ``ssm``: when a step improves the objective by less
    than ``m_step_tolerance`` relative to its magnitude (or to 1), or the largest gradient entry falls below
    ``m_step_tolerance``, or after ``m_step_num_iters`` steps. Any other ``optax.GradientTransformation`` is
    accepted and runs for ``m_step_num_iters`` steps. Frozen leaves (``trainable=False``) are held fixed; freezing ``recurrent_weights`` at zero gives
    ``ssm``'s ``InputDrivenTransitions``.

    :param num_states: :math:`K`
    :param state_dim: :math:`D`
    :param input_dim: :math:`M`
    :param recurrent_only: drop the :math:`K \times K` matrix for a single bias row
    :param concentration: Dirichlet concentration :math:`\beta` of the rows of :math:`\mathrm{softmax}(\mathrm{logits})`
    :param stickiness: extra concentration :math:`\kappa` on the diagonal
    :param l2_penalty_recurrent: :math:`\lambda_x`
    :param l2_penalty_input: :math:`\lambda_u`
    :param m_step_optimizer: an ``optax`` optimizer
    :param m_step_num_iters: at most this many optimizer steps per M-step
    :param m_step_tolerance: the L-BFGS stopping tolerance on the objective's relative improvement and on the gradient
    """
    depends_on_x = True

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 input_dim: int = 0,
                 recurrent_only: bool = False,
                 concentration: Union[Scalar, Array] = 1.0,
                 stickiness: Union[Scalar, Array] = 0.0,
                 l2_penalty_recurrent: float = 0.0,
                 l2_penalty_input: float = 0.0,
                 m_step_optimizer: Optional[optax.GradientTransformation] = None,
                 m_step_num_iters: int = 1000,
                 m_step_tolerance: float = 1e-6):
        self.num_states = num_states
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.recurrent_only = recurrent_only
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)
        self.l2_penalty_recurrent = l2_penalty_recurrent
        self.l2_penalty_input = l2_penalty_input
        self.m_step_optimizer = optax.lbfgs() if m_step_optimizer is None else optax.with_extra_args_support(m_step_optimizer)
        self.m_step_num_iters = m_step_num_iters
        self.m_step_tolerance = m_step_tolerance

    def initialize(self,
                   key: Optional[PRNGKeyT] = None,
                   method: str = "prior",
                   logits: Optional[Array] = None,
                   recurrent_weights: Optional[Array] = None,
                   input_weights: Optional[Array] = None
                   ) -> Tuple[ParamsRecurrentSLDSTransitions, ParamsRecurrentSLDSTransitions]:
        """Parameters and properties.

        ``method="prior"``: ``logits = log P`` with :math:`P` a Dirichlet draw per row, zero recurrent
        weights and standard normal input weights (``recurrent_only``: standard normal logits, recurrent
        weights and input weights, as ``ssm``). ``method="sticky"`` (the ARHMM initialization's start, as
        ``ssm``): ``logits = log P`` with :math:`P = 0.95 I + 0.05 U` row-normalized (``recurrent_only``:
        zeros), zero recurrent weights and standard normal input weights (``ssm``'s ``InputDrivenTransitions``).
        """
        K, D, M = self.num_states, self.state_dim, self.input_dim
        if key is None:
            if logits is None or recurrent_weights is None or input_weights is None:
                raise ValueError("key must be provided if any parameter is not provided.")
            key = jr.PRNGKey(0)                        # nothing drawn from it is used
        k1, k2, k3 = jr.split(key, 3)
        if method == "prior":
            if self.recurrent_only:
                default_logits = jr.normal(k1, (1, K))
                default_recurrent = jr.normal(k2, (K, D))
                default_inputs = jr.normal(k3, (K, M))
            else:
                default_logits = jnp.log(jr.dirichlet(k1, self.concentration))
                default_recurrent, default_inputs = jnp.zeros((K, D)), jr.normal(k3, (K, M))
        elif method == "sticky":
            if self.recurrent_only:
                default_logits = jnp.zeros((1, K))
            else:
                P = 0.95 * jnp.eye(K) + 0.05 * jr.uniform(k1, (K, K))
                default_logits = jnp.log(P / P.sum(axis=1, keepdims=True))
            default_recurrent, default_inputs = jnp.zeros((K, D)), jr.normal(k3, (K, M))
        else:
            raise ValueError(f"Unknown initialization method {method!r}")
        params = ParamsRecurrentSLDSTransitions(
            logits=default_logits if logits is None else logits,
            recurrent_weights=default_recurrent if recurrent_weights is None else recurrent_weights,
            input_weights=default_inputs if input_weights is None else input_weights)
        props = ParamsRecurrentSLDSTransitions(logits=ParameterProperties(), recurrent_weights=ParameterProperties(),
                                               input_weights=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsRecurrentSLDSTransitions) -> Scalar:
        """Dirichlet rows (unless ``recurrent_only`` or K = 1) plus the L2 penalties."""
        lp = -0.5 * self.l2_penalty_recurrent * jnp.sum(params.recurrent_weights ** 2) \
             - 0.5 * self.l2_penalty_input * jnp.sum(params.input_weights ** 2)
        if not self.recurrent_only and self.num_states > 1:
            lp += tfd.Dirichlet(self.concentration).log_prob(jax.nn.softmax(params.logits, axis=-1)).sum()
        return lp

    def _unnormalized_logits(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array) -> Array:
        """(T-1, K, K): logits[i, j] + W_x[j]·x_t + W_u[j]·u_{t+1} for the pair t -> t+1."""
        effect = x[:-1] @ params.recurrent_weights.T + inputs[1:] @ params.input_weights.T    # (T-1, K)
        logits = params.logits[None] + effect[:, None, :]                                     # (T-1, 1|K, K)
        return jnp.broadcast_to(logits, (x.shape[0] - 1, self.num_states, self.num_states))

    def log_transition_matrices(self, params, x, inputs) -> Array:
        return jax.nn.log_softmax(self._unnormalized_logits(params, x, inputs), axis=-1)

    def transition_matrices(self, params, x, inputs) -> Array:
        return jax.nn.softmax(self._unnormalized_logits(params, x, inputs), axis=-1)

    def transition_row(self, params, state, x, next_input) -> Array:
        row = params.logits[0 if self.recurrent_only else state]
        return jax.nn.softmax(row + params.recurrent_weights @ x + params.input_weights @ next_input)

    def expected_log_trans_prob(self, params, x, inputs, trans_probs) -> Scalar:
        """sum_t sum_ij Ezzp1[t, i, j] log P_t(x_t, u_{t+1})[i, j]."""
        return jnp.sum(trans_probs * self.log_transition_matrices(params, x, inputs))

    def neg_hessian_x(self, params, x, inputs, trans_probs) -> Array:
        r"""(T-1, D, D): :math:`\sum_i \mathbb{E}[z_t = i]\, \mathrm{Cov}_{P_t[i, :]}(W_x)`, the negative Hessian in
        :math:`x_t` of the expected log transition term (the log-softmax is concave in :math:`x_t`)."""
        P = self.transition_matrices(params, x, inputs)                                      # (T-1, K, K)
        W = params.recurrent_weights                                                         # (K, D)
        Ez = trans_probs.sum(axis=2)                                                         # (T-1, K): E[z_t = i]
        second = jnp.einsum("tij,jd,je->tide", P, W, W)                                      # E_j[W_j W_j^T] per row i
        mean = jnp.einsum("tij,jd->tid", P, W)                                               # E_j[W_j] per row i
        cov = second - jnp.einsum("tid,tie->tide", mean, mean)
        return jnp.einsum("ti,tide->tde", Ez, cov)

    def collect_suff_stats(self, params, qz, x_samples, inputs) -> RecurrentTransStats:
        """The pairwise expectations with the samples of x and the inputs the M-step objective needs."""
        return RecurrentTransStats(trans_probs=qz.trans_probs, x_samples=x_samples, inputs=inputs)

    def initialize_m_step_state(self, params, props) -> Any:
        """The optimizer state is re-initialized at every M-step, so nothing is carried."""
        return None

    def m_step(self, params, props, batch_stats: RecurrentTransStats, m_step_state) -> Tuple[ParamsRecurrentSLDSTransitions, Any]:
        r"""Numerically maximize :math:`\log p(\theta) + \sum_b \frac{1}{S} \sum_s \sum_t \sum_{ij} \mathbb{E}[z_t = i, z_{t+1} = j] \log P_t(x^{(s)}_{b,t}, u_{b,t+1})_{ij}`,
        scaled by :math:`1 / (B T)`, from the current parameters. With a line-search optimizer (``optax.lbfgs``)
        the loop stops by ``m_step_tolerance`` as described in the class docstring; otherwise it runs
        ``m_step_num_iters`` steps."""
        is_leaf = lambda node: isinstance(node, ParameterProperties)
        trainable = tree_map(lambda p: p.trainable, props, is_leaf=is_leaf)
        if not any(jax.tree_util.tree_leaves(trainable)):
            return params, m_step_state
        fixed = params
        rebuild = lambda free: tree_map(lambda f, p0, t: f if t else p0, free, fixed, trainable)
        num_sequences, num_timesteps = batch_stats.inputs.shape[:2]
        scale = num_sequences * num_timesteps

        def objective(free):
            current = rebuild(free)

            def _one_sequence(stats):
                per_sample = vmap(lambda x: jnp.sum(stats.trans_probs * self.log_transition_matrices(current, x, stats.inputs)))
                return per_sample(stats.x_samples).mean()

            return -(self.log_prior(current) + vmap(_one_sequence)(batch_stats).sum()) / scale

        optimizer = self.m_step_optimizer
        state = optimizer.init(params)
        tol, max_steps = self.m_step_tolerance, self.m_step_num_iters
        # a line-search optimizer stores the value and gradient at the accepted point, so the stopping test
        # costs nothing; without them, run the fixed number of steps (as ssm does for its SGD-type optimizers)
        line_search = otu.tree_get(state, "value") is not None and otu.tree_get(state, "grad") is not None
        if line_search:
            value_and_grad = optax.value_and_grad_from_state(objective)
        else:
            plain = jax.value_and_grad(objective)
            value_and_grad = lambda free, state: plain(free)

        def _cond(carry):
            _, _, num_steps, done = carry
            return (num_steps == 0) | ((num_steps < max_steps) & ~done)

        def _body(carry):
            free, state, num_steps, _ = carry
            value, grad = value_and_grad(free, state=state)
            updates, state = optimizer.update(grad, state, free, value=value, grad=grad, value_fn=objective)
            free = optax.apply_updates(free, updates)
            if line_search:
                new_value, new_grad = otu.tree_get(state, "value"), otu.tree_get(state, "grad")
                improvement = (value - new_value) / jnp.maximum(jnp.maximum(jnp.abs(value), jnp.abs(new_value)), 1.0)
                largest_gradient = jax.tree_util.tree_reduce(jnp.maximum, tree_map(lambda g: jnp.max(jnp.abs(g), initial=0.0), new_grad))
                done = (improvement <= tol) | (largest_gradient <= tol)
            else:
                done = False
            return free, state, num_steps + 1, done

        free, _, _, _ = lax.while_loop(_cond, _body, (params, state, 0, False))
        return rebuild(free), m_step_state

    def as_hmm_transitions(self, state_dim: int):
        """The ``HMMTransitions`` adapter used by the ARHMM initialization."""
        from dynamax.slds.initialization import _ARHMMTransitions
        return _ARHMMTransitions(self, state_dim)
