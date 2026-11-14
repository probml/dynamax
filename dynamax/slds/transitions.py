"""
Transitions between the discrete states of a switching linear dynamical system (SLDS).

:class:`StationarySLDSTransitions` uses a fixed transition matrix, as in the standard SLDS.
:class:`RecurrentSLDSTransitions` makes the probabilities of the next discrete state depend on the current
continuous state and the next input, as in the recurrent SLDS (rSLDS).
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
    """A fixed transition matrix with a Dirichlet prior on each row.

    :param num_states: number of discrete states
    :param concentration: Dirichlet concentration of each row; the default of 1.0 gives the maximum likelihood update
    :param stickiness: extra concentration on the diagonal, which favors staying in the same state
    """
    depends_on_x = False

    def __init__(self, num_states: int, concentration: Union[Scalar, Array] = 1.0, stickiness: Union[Scalar, Array] = 0.0):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)

    def initialize(self, key: Optional[PRNGKeyT] = None, method: str = "prior",
                   transition_matrix: Optional[Array] = None) -> Tuple[ParamsStationarySLDSTransitions, ParamsStationarySLDSTransitions]:
        """Initialize the transition matrix.

        Args:
            key: random number generator; required unless ``transition_matrix`` is given
            method: ``"prior"`` draws each row from the Dirichlet prior; ``"sticky"`` draws a random
                matrix with most of its weight on the diagonal
            transition_matrix: optional transition matrix, shape (K, K), to use instead of a random one

        Returns:
            tuple of parameters and their properties
        """
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
        """Compute the log prior of the transition matrix; it is zero when there is a single state."""
        if self.num_states == 1:
            return 0.0
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    def transition_matrices(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array) -> Array:
        """Return the transition matrix, shape (K, K)."""
        return params.transition_matrix

    def log_transition_matrices(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array) -> Array:
        """Return the log of the transition matrix, shape (K, K)."""
        return jnp.log(params.transition_matrix)

    def transition_row(self, params: ParamsStationarySLDSTransitions, state: Array, x: Array, next_input: Array) -> Array:
        """Return the probabilities of the next discrete state given the current one, shape (K,)."""
        return params.transition_matrix[state]

    def expected_log_trans_prob(self, params: ParamsStationarySLDSTransitions, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """Compute the expected log probability of the transitions under the discrete posterior.

        ``trans_probs`` holds the posterior probability of each pair of consecutive states, summed over time.
        """
        return jnp.sum(xlogy(trans_probs, params.transition_matrix))

    def collect_suff_stats(self, params: ParamsStationarySLDSTransitions, qz: DiscretePosteriorSLDS, x_samples: Array,
                           inputs: Array) -> Array:
        """Return the expected number of transitions between each pair of states, shape (K, K)."""
        return qz.trans_probs

    def initialize_m_step_state(self, params: ParamsStationarySLDSTransitions, props: ParamsStationarySLDSTransitions) -> Any:
        return None

    def m_step(self,
               params: ParamsStationarySLDSTransitions,
               props: ParamsStationarySLDSTransitions,
               batch_stats: Array,
               m_step_state: Any) -> Tuple[ParamsStationarySLDSTransitions, Any]:
        """Update the transition matrix to the mode of its posterior given the expected counts.

        The counts plus the prior are clipped at zero, and a tiny floor relative to each row's total is added so
        that no transition probability is exactly zero.
        """
        if props.transition_matrix.trainable:
            counts = jnp.maximum(batch_stats.sum(axis=0) + (self.concentration - 1.0), 0.0)
            counts = counts + 1e-32 * jnp.maximum(counts.sum(axis=-1, keepdims=True), 1.0)
            transition_matrix = counts / counts.sum(axis=-1, keepdims=True)
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state


class ParamsRecurrentSLDSTransitions(NamedTuple):
    """Parameters of the recurrent transitions of an SLDS.

    :param logits: transition logits, shape (K, K), or (1, K) when ``recurrent_only``
    :param recurrent_weights: weights on the continuous state, shape (K, D)
    :param input_weights: weights on the inputs, shape (K, M)
    """
    logits: Union[Float[Array, "num_states num_states"], Float[Array, "1 num_states"], ParameterProperties]
    recurrent_weights: Union[Float[Array, "num_states state_dim"], ParameterProperties]
    input_weights: Union[Float[Array, "num_states input_dim"], ParameterProperties]


class RecurrentSLDSTransitions(SLDSTransitions):
    """Transitions whose probabilities depend on the current continuous state and the next input.

    These are the transitions of the recurrent SLDS (rSLDS) of Linderman et al. (2017). The logits of the next
    discrete state are the row of ``logits`` for the current discrete state, plus a linear function of the
    continuous state, plus a linear function of the next input.

    The M-step has no closed form, so it runs an optimizer starting from the current parameters. With a
    line-search optimizer such as the default L-BFGS, it stops when a step barely improves the objective or the
    gradient is close to zero (both judged by ``m_step_tolerance``), or after ``m_step_num_iters`` steps. Other
    ``optax`` optimizers run for ``m_step_num_iters`` steps. Parameters marked as not trainable are held fixed; freezing
    ``recurrent_weights`` at zero gives transitions that depend on the inputs only.

    :param num_states: number of discrete states
    :param state_dim: dimension of the continuous state
    :param input_dim: dimension of the inputs
    :param recurrent_only: whether to replace the (K, K) logits with a single row shared by all discrete states,
        so the next state depends on the continuous state and inputs but not on the current discrete state
    :param concentration: Dirichlet concentration of each row of the transition matrix given by the logits;
        not used when ``recurrent_only``
    :param stickiness: extra Dirichlet concentration on the diagonal, which favors staying in the same state
    :param l2_penalty_recurrent_weights: strength of the L2 penalty on ``recurrent_weights``
    :param l2_penalty_input_weights: strength of the L2 penalty on ``input_weights``
    :param m_step_optimizer: ``optax`` optimizer for the M-step; defaults to L-BFGS
    :param m_step_num_iters: maximum number of optimizer steps per M-step
    :param m_step_tolerance: stopping tolerance of a line-search optimizer, as described above
    """
    depends_on_x = True

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 input_dim: int = 0,
                 recurrent_only: bool = False,
                 concentration: Union[Scalar, Array] = 1.0,
                 stickiness: Union[Scalar, Array] = 0.0,
                 l2_penalty_recurrent_weights: float = 0.0,
                 l2_penalty_input_weights: float = 0.0,
                 m_step_optimizer: Optional[optax.GradientTransformation] = None,
                 m_step_num_iters: int = 1000,
                 m_step_tolerance: float = 1e-6):
        self.num_states = num_states
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.recurrent_only = recurrent_only
        self.concentration = concentration * jnp.ones((num_states, num_states)) + stickiness * jnp.eye(num_states)
        self.l2_penalty_recurrent_weights = l2_penalty_recurrent_weights
        self.l2_penalty_input_weights = l2_penalty_input_weights
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
        """Initialize the transition parameters.

        With ``method="prior"``, the logits are the log of a transition matrix whose rows are drawn from the
        Dirichlet prior, the recurrent weights are zero and the input weights are standard normal. When
        ``recurrent_only``, the logits, recurrent weights and input weights are all standard normal instead.

        With ``method="sticky"``, the logits are the log of a random transition matrix with most of its weight
        on the diagonal (zeros when ``recurrent_only``), the recurrent weights are zero and the input weights are
        standard normal.

        Args:
            key: random number generator; required unless all three parameters are given
            method: ``"prior"`` or ``"sticky"``, as described above
            logits: optional logits, shape (K, K), or (1, K) when ``recurrent_only``
            recurrent_weights: optional weights on the continuous state, shape (K, D)
            input_weights: optional weights on the inputs, shape (K, M)

        Returns:
            tuple of parameters and their properties
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
        """Compute the log prior of the parameters.

        This is the L2 penalties on the weights plus, unless ``recurrent_only`` or there is a single state, the
        Dirichlet log prior of the rows of the transition matrix given by the logits.
        """
        lp = -0.5 * self.l2_penalty_recurrent_weights * jnp.sum(params.recurrent_weights ** 2) \
             - 0.5 * self.l2_penalty_input_weights * jnp.sum(params.input_weights ** 2)
        if not self.recurrent_only and self.num_states > 1:
            lp += tfd.Dirichlet(self.concentration).log_prob(jax.nn.softmax(params.logits, axis=-1)).sum()
        return lp

    def _unnormalized_logits(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array,
                             broadcast: bool = True) -> Array:
        """Logits of the transition from each time step to the next, shape (T-1, K, K).

        Each entry uses the continuous state at the earlier of its two time steps and the input at the later one.
        Without ``broadcast``, the shape is (T-1, 1, K) when ``recurrent_only``.
        """
        effect = x[:-1] @ params.recurrent_weights.T + inputs[1:] @ params.input_weights.T    # (T-1, K)
        logits = params.logits[None] + effect[:, None, :]                                     # (T-1, 1|K, K)
        if not broadcast:
            return logits
        return jnp.broadcast_to(logits, (x.shape[0] - 1, self.num_states, self.num_states))

    def log_transition_matrices(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array) -> Array:
        return jax.nn.log_softmax(self._unnormalized_logits(params, x, inputs), axis=-1)

    def transition_matrices(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array) -> Array:
        return jax.nn.softmax(self._unnormalized_logits(params, x, inputs), axis=-1)

    def transition_row(self, params: ParamsRecurrentSLDSTransitions, state: Array, x: Array, next_input: Array) -> Array:
        row = params.logits[0 if self.recurrent_only else state]
        return jax.nn.softmax(row + params.recurrent_weights @ x + params.input_weights @ next_input)

    def expected_log_trans_prob(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array, trans_probs: Array) -> Scalar:
        """Compute the expected log probability of the transitions under the discrete posterior, given ``x``."""
        return jnp.sum(trans_probs * self.log_transition_matrices(params, x, inputs))

    def neg_hessian_x(self, params: ParamsRecurrentSLDSTransitions, x: Array, inputs: Array, trans_probs: Array) -> Array:
        """Compute the negative Hessian of the expected log transition probability in each continuous state.

        The result has shape (T-1, D, D). At each time step it is the covariance of the recurrent weights under the
        next-state probabilities, averaged over the posterior of the current discrete state. It is positive
        semidefinite.
        """
        P = self.transition_matrices(params, x, inputs)                                      # (T-1, K, K)
        W = params.recurrent_weights                                                         # (K, D)
        smoothed_probs = trans_probs.sum(axis=2)                                             # (T-1, K): E[z_t = i]
        second = jnp.einsum("tij,jd,je->tide", P, W, W)                                      # E_j[W_j W_j^T] per row i
        mean = jnp.einsum("tij,jd->tid", P, W)                                               # E_j[W_j] per row i
        cov = second - jnp.einsum("tid,tie->tide", mean, mean)
        return jnp.einsum("ti,tide->tde", smoothed_probs, cov)

    def collect_suff_stats(self, params: ParamsRecurrentSLDSTransitions, qz: DiscretePosteriorSLDS, x_samples: Array,
                           inputs: Array) -> RecurrentTransStats:
        """Collect the posterior transition probabilities, continuous-state samples and inputs for the M-step."""
        return RecurrentTransStats(trans_probs=qz.trans_probs, x_samples=x_samples, inputs=inputs)

    def initialize_m_step_state(self, params: ParamsRecurrentSLDSTransitions, props: ParamsRecurrentSLDSTransitions) -> Any:
        """Return ``None``; the optimizer starts afresh at every M-step, so no state is carried between them."""
        return None

    def m_step(self,
               params: ParamsRecurrentSLDSTransitions,
               props: ParamsRecurrentSLDSTransitions,
               batch_stats: RecurrentTransStats,
               m_step_state: Any) -> Tuple[ParamsRecurrentSLDSTransitions, Any]:
        """Update the parameters by numerical optimization, starting from the current parameters.

        The objective is the log prior plus the expected log probability of the transitions, averaged over the
        samples of the continuous states and summed over sequences, then divided by the number of sequences times
        the number of time steps. It stops as described in the class docstring, after at least one step.
        """
        is_leaf = lambda node: isinstance(node, ParameterProperties)
        trainable = tree_map(lambda p: p.trainable, props, is_leaf=is_leaf)
        if not any(jax.tree_util.tree_leaves(trainable)):
            return params, m_step_state
        fixed = params
        rebuild = lambda free: tree_map(lambda f, p0, t: f if t else p0, free, fixed, trainable)
        num_sequences, num_timesteps = batch_stats.inputs.shape[:2]
        scale = num_sequences * num_timesteps
        if self.recurrent_only:
            # every row of a log transition matrix is the same, so sum the pair probabilities over the earlier state
            batch_stats = batch_stats._replace(trans_probs=batch_stats.trans_probs.sum(axis=-2, keepdims=True))

        def objective(free):
            current = rebuild(free)

            def _one_sequence(stats):
                log_P = lambda x: jax.nn.log_softmax(self._unnormalized_logits(current, x, stats.inputs, broadcast=False), axis=-1)
                per_sample = vmap(lambda x: jnp.sum(stats.trans_probs * log_P(x)))
                return per_sample(stats.x_samples).mean()

            return -(self.log_prior(current) + vmap(_one_sequence)(batch_stats).sum()) / scale

        optimizer = self.m_step_optimizer
        state = optimizer.init(params)
        tol, max_steps = self.m_step_tolerance, self.m_step_num_iters
        # a line-search optimizer stores the value and gradient at the accepted point, so the stopping test
        # costs nothing; without them, run the fixed number of steps
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
