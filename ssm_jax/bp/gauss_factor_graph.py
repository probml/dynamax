import functools
from functools import partial
from typing import NamedTuple, Union, List, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import register_pytree_node_class, tree_leaves, tree_map
from ssm_jax.bp.gauss_bp_utils import (info_divide,
                                       info_marginalize,
                                       info_multiply)

def _tree_reduce(function, tree, initializer=None, is_leaf=None):
    """Copy of jax.tree_utils.reduce which accepts an `is_leaf` argument."""
    if initializer is None:
        return functools.reduce(function, tree_leaves(tree, is_leaf))
    else:
        return functools.reduce(function, tree_leaves(tree, is_leaf), initializer)

class CanonicalPotential(NamedTuple):
    """Container class for a Canonical Potential.

    eta: (N,) array.
    Lambda: (N, N) array.
    """
    eta: jnp.ndarray
    Lambda: jnp.ndarray

def zeros_canonical_pot(dim):
    """Construct a Canonical potential with all entries zero."""
    eta = jnp.zeros(dim)
    Lambda = jnp.zeros((dim, dim))
    return CanonicalPotential(eta=eta, Lambda=Lambda)


@jit
def sum_reduce_cpots(cpots, initializer=None):
    """Sum over the corresponding parameters for potential in `cpots`."""
    return _tree_reduce(info_multiply, cpots, initializer,
                        is_leaf=lambda l: isinstance(l, CanonicalPotential))


class GaussianVariableNode(NamedTuple):
    """A Gaussian variable node in a factor graph."""
    varID : Union[int, str]
    dim : int
    prior : CanonicalPotential # Currently using prior to 'roll in' single variable factors.

class CanonicalFactor(NamedTuple):
    """A canonical factor involving `len(adj_varIDs)` variables."""
    # TODO: Not totally clear what the best way is to handle the var_scopes.
    # Could use a dataclass with a __post_init__ to construct automatically...
    #  I'm not sure how this plays with rolling/unrolling...
    factorID : Union[int, str]
    adj_varIDs : Union[List[int], List[str]]
    potential : CanonicalPotential
    var_scopes : Dict # {varID : (var_start, var_stop)}

def make_canonical_factor(factorID, var_nodes, cpot):
    """ Helper function to construct a Canonical Factor."""
    # It should be possible to replace this with the right constructor for CanonicalFactor
    varIDs = [var.varID for var in var_nodes]
    var_scopes = _calculate_var_scopes(var_nodes)
    return CanonicalFactor(factorID, varIDs, cpot, var_scopes)

class GaussianFactorGraph(NamedTuple):
    """ A container for variables and factor in a factor graph."""
    var_nodes : List[GaussianVariableNode]
    factors : List[CanonicalFactor]
    # factor_to_var_edges are already implicitly contained factors... 
    factor_to_var_edges : Dict
    var_to_factor_edges : Dict

def make_factor_graph(var_nodes, factors):
    """Helper function to construct a factor graph."""
    # TODO: Another place where maybe this belongs as a constructor in the GFG class.
    factor_to_var_edges = {factor.factorID: factor.adj_varIDs for factor in factors}
    var_to_factor_edges = {var.varID: [fID for fID, vIDs in factor_to_var_edges.items() if var.varID in vIDs]
                           for var in var_nodes}
    return GaussianFactorGraph(var_nodes,factors,
                               factor_to_var_edges,
                               var_to_factor_edges)

def init_messages(factor_graph):
    """ Initial factor to variable messages as zeros."""
    var_dims = {var.varID : var.dim for var in factor_graph.var_nodes}
    return {factor.factorID : {vID: zeros_canonical_pot(var_dims[vID]) for vID in factor.adj_varIDs}
            for factor in factor_graph.factors}

@partial(jit, static_argnums=2)
def absorb_canonical_message(cpot, message, message_scope):
    """Absorb a canonical message into a potential."""
    var_start, var_stop = message_scope
    eta = cpot.eta.at[var_start:var_stop].add(message.eta)
    Lambda = cpot.Lambda.at[var_start:var_stop, var_start:var_stop].add(message.Lambda)
    return CanonicalPotential(eta, Lambda)

def absorb_var_to_factor_messages(factor, messages):
    """Absorb all the messages into a factor potential."""
    pot = tree_map(jnp.copy,factor.potential)
    for varID, message in messages.items():
        pot = absorb_canonical_message(pot, message, factor.var_scopes[varID])
    return pot

def marginalise_onto_var(potential, var_scope):
    """Marginalise a joint canonical form Gaussian onto a variable."""
    (K11, K12, K22), (h1, h2) = extract_canonical_potential_blocks(potential, var_scope)
    K_marg, h_marg = info_marginalize(K22, K12.T, K11, h2, h1)
    return CanonicalPotential(eta=h_marg, Lambda=K_marg)

def update_belief(var, messages):
    """Combine incoming messages to calculate a variable's belief state."""
    return sum_reduce_cpots(messages, initializer=var.prior)

def calculate_var_belief(var,messages, var_to_factor_edges):
    """Combine incoming messages to calculate a variable's belief state."""
    incoming_messages = [messages[f][var.varID] for f in var_to_factor_edges[var.varID]]
    return update_belief(var, incoming_messages)

def calculate_all_beliefs(factor_graph, messages):
    """Absorb messages for all variables to calculate all belief states."""
    return {var.varID: calculate_var_belief(var, messages, factor_graph.var_to_factor_edges)
            for var in factor_graph.var_nodes}

def update_factor_to_var_messages(factor,var_beliefs,messages_to_vars,damping=0):
    """Given current variable belief state and the previous messages sent to each variable
     calculate the update messages to send to variables.
     
     Note: Ordinarily the message sent to the ith variable, var_i, is given by summing the incoming 
      messages of all *other* variables and then marginalising onto var_i.

      In Canonical Gaussian form we can sum *all* messages and marginalise onto var_i before simply 
       substracting the message from var_i. This trick allows us perform a single message absorption  
       step instead of repeating for each variable.

    Args:
        factor: A CanonicalFactor object.
        var_beliefs: A dictionary `{varID: CanonicalPotential}` containing marginal variable belief 
                     states.
        messages_to_vars: A dictionary {varID: CanonicalPotential} containing messages from `factor`
                          to each variable from the previous step.
        damping: float.

    Returns:
        messages_to_vars: A dictionary {varID: CanonicalPotential} containing the updated messages 
                           from `factor` to each variable.
      """
    # Divide var.beliefs by messages --> message_v_to_f.
    var_messages = info_divide(var_beliefs, messages_to_vars)
    # Absorb all messages into factor.
    pot_plus_messages = absorb_var_to_factor_messages(factor, var_messages)
    for vID in factor.adj_varIDs:
        # Marginalise message_v_to_f onto var.
        var_marginal = marginalise_onto_var(pot_plus_messages, factor.var_scopes[vID])
        raw_message = info_divide(var_marginal, var_messages[vID]) # Substract message_f_to_v
        damped_message = jax.tree_map(
            lambda x, y: damping * x + (1 - damping) * y,
            messages_to_vars[vID], raw_message
        )
        messages_to_vars[vID] = damped_message
    return messages_to_vars

def update_all_messages(factor_graph, messages, damping=0):
    """Loop over all factors in the graph calculated updated messages to variables.
   
    Args:
        factor_graph: A GaussianFactorGraph object.
        messages: a dict of dicts containing messages for each factor -
                 {factorID : {varID: message, ...}, ...}
    
    Returns:
        new_messages: dict with the same form as `messages` containing updated factor to variable
                       messages.
    """
    var_beliefs = calculate_all_beliefs(factor_graph, messages)
    new_messages = {}
    for factor in factor_graph.factors:
        factor_var_beliefs = {vID:var_beliefs[vID] for vID in factor.adj_varIDs}
        messages_to_vars = messages[factor.factorID]
        new_messages[factor.factorID] = update_factor_to_var_messages(
            factor, factor_var_beliefs, messages_to_vars, damping
            )
    return new_messages

def extract_canonical_potential_blocks(can_pot, var_scope):
    """Split a precision matrix into blocks for marginalising / conditionalising.

    E.g. K = [[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24]]
        h = [0, 1, 2, 3, 4, 5]
        idxs = [1,2]
        gets split into:
            K11 - [[ 6,  7],
                   [11, 12]]

            K12 - [[ 5,  8,  9],
                   [10, 13, 14]]

            K22 - [[ 0,  3,  4],
                   [15, 18, 19],
                   [20, 23, 24]]
        and
            h1 - [1, 2]
            h2 - [0, 3, 4]

    Args:
        can_pot - a CanonicalPotential object (or similar tuple) with elements (h, K) where,
                    K - (D x D) precision matrix.
                    h - (D,) potential vector.
        idxs (N,) array of indices in 1,...,D.
    Returns:
        (K11, K12, K22), (h1, h2) - blocks of the potential parameters where:
            K11 (N x N) block of precision elements with row and column in `indxs`
            K12 (N x D-N) block of precision elements with row in `indxs` but column not in `indxs`.
            K22 (D-N x D-N) block of precision elements with neither row nor column in `indxs`.
            h1 (N,) elements of potential vector in `indxs`.
            h2 (D-N,) elements of potential vector not in `indxs`.
    """
    # TODO: Investigate using jax.lax.dynamic_slice instead.
    # TODO: also maybe precompute the ~b indices.
    h, K = can_pot
    # Using np instead of jnp so that these aren't traced.
    idxs = np.arange(*var_scope)
    idx_range = np.arange(len(h))
    b = np.isin(idx_range, idxs)
    K11 = K[b, :][:, b]
    K12 = K[b, :][:, ~b]
    K22 = K[~b, :][:, ~b]
    h1 = h[b]
    h2 = h[~b]
    return (K11, K12, K22), (h1, h2)


def _calculate_var_scopes(var_nodes):
    """Helper function to calculate the variable scopes for a factor involving `var_nodes`."""
    var_ids = np.array([var.varID for var in var_nodes])
    var_dims = np.array([var.dim for var in var_nodes], dtype=np.int32)
    # Use numpy so it is not traced.
    var_starts = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(var_dims)[:-1]))
    var_stops = var_starts + var_dims
    var_scopes = {var_id: (start, stop)
                    for var_id, start, stop in zip(var_ids, var_starts, var_stops)}
    return var_scopes