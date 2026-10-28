r"""
Switching linear dynamical systems with linear Gaussian dynamics and emissions.
"""
from typing import Optional

from dynamax.slds.abstractions import SLDS, SLDSInitialState
from dynamax.slds.dynamics import LinearGaussianSLDSDynamics
from dynamax.slds.emissions import LinearGaussianSLDSEmissions
from dynamax.slds.transitions import StationarySLDSTransitions


class GaussianSLDS(SLDS):
    r"""Switching linear dynamical system with a fixed transition matrix (``ssm``'s ``SLDS`` with
    ``transitions="standard"`` or ``"sticky"``, ``dynamics="gaussian"``, ``emissions="gaussian"``).

    .. math::
        z_t \mid z_{t-1} \sim \mathrm{Cat}(P[z_{t-1}, :]), \quad
        x_t \mid x_{t-1}, z_t \sim \mathcal{N}(F_{z_t} x_{t-1} + b_{z_t} + B_{z_t} u_t, Q_{z_t}), \quad
        y_t \mid x_t, z_t \sim \mathcal{N}(H_{z_t} x_t + d_{z_t} + D_{z_t} u_t, R_{z_t})

    :param num_states: :math:`K`
    :param state_dim: :math:`D`
    :param emission_dim: :math:`N`
    :param input_dim: :math:`M` (0 for no inputs)
    :param single_subspace: one emission model shared by all states (default) or one per state
    :param dynamics_cov_type: ``"full"`` (default) or ``"diag"`` :math:`Q_k`
    :param emission_cov_type: ``"diag"`` (default, as ``ssm``) or ``"full"`` :math:`R_k`
    :param initial_probs_concentration: Dirichlet concentration of :math:`\pi_0`
    :param transition_matrix_concentration: Dirichlet concentration of the rows of :math:`P`
    :param transition_matrix_stickiness: extra concentration on the diagonal of :math:`P`
    :param dynamics_kwargs: hyperparameters of :class:`LinearGaussianSLDSDynamics` (ridges, ``nu0``, ``Psi0``)
    :param emission_kwargs: hyperparameters of :class:`LinearGaussianSLDSEmissions` (``l2_penalty``, ``nu0``, ``Psi0``)
    :param driver_kwargs: see :class:`SLDS` (``num_inner_sweeps``, ``damping``, ``newton_tolerance``, ...)
    """

    def __init__(self,
                 num_states: int,
                 state_dim: int,
                 emission_dim: int,
                 input_dim: int = 0,
                 single_subspace: bool = True,
                 dynamics_cov_type: str = "full",
                 emission_cov_type: str = "diag",
                 initial_probs_concentration: float = 1.0,
                 transition_matrix_concentration: float = 1.0,
                 transition_matrix_stickiness: float = 0.0,
                 dynamics_kwargs: Optional[dict] = None,
                 emission_kwargs: Optional[dict] = None,
                 **driver_kwargs):
        self.emission_dim = emission_dim
        super().__init__(
            num_states, state_dim,
            initial_component=SLDSInitialState(num_states, initial_probs_concentration),
            transition_component=StationarySLDSTransitions(num_states, transition_matrix_concentration, transition_matrix_stickiness),
            dynamics_component=LinearGaussianSLDSDynamics(num_states, state_dim, input_dim, dynamics_cov_type, **(dynamics_kwargs or {})),
            emission_component=LinearGaussianSLDSEmissions(num_states, state_dim, emission_dim, input_dim, single_subspace,
                                                           emission_cov_type, **(emission_kwargs or {})),
            **driver_kwargs)

