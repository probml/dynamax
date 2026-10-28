"""
Switching linear dynamical systems with linear Gaussian dynamics and emissions.
"""
from typing import Optional

from dynamax.slds.abstractions import SLDS, SLDSInitialState
from dynamax.slds.dynamics import LinearGaussianSLDSDynamics
from dynamax.slds.emissions import LinearGaussianSLDSEmissions
from dynamax.slds.transitions import StationarySLDSTransitions


class GaussianSLDS(SLDS):
    r"""Switching linear dynamical system with a fixed transition matrix and Gaussian dynamics and emissions.

    The model is defined as follows

    $$z_1 \sim \mathrm{Cat}(\pi_0)$$
    $$z_t \mid z_{t-1} \sim \mathrm{Cat}(P_{z_{t-1}})$$
    $$x_1 \sim \mathcal{N}(m_0, \Sigma_0)$$
    $$x_t \mid x_{t-1}, z_t, u_t \sim \mathcal{N}(F_{z_t} x_{t-1} + B_{z_t} u_t + b_{z_t}, Q_{z_t})$$
    $$y_t \mid x_t, z_t, u_t \sim \mathcal{N}(H_{z_t} x_t + D_{z_t} u_t + d_{z_t}, R_{z_t})$$

    where

    * $z_t \in \{1, \ldots, K\}$ is a discrete latent state,
    * $x_t$ is a continuous latent state of size `state_dim`,
    * $y_t$ is an emission of size `emission_dim`,
    * $u_t$ is an input of size `input_dim` (defaults to 0),
    * $\pi_0$ and $P$ are the initial distribution and the transition matrix of the discrete state,
    * $m_0$ and $\Sigma_0$ are the mean and covariance of the initial continuous state,
    * $F_k$, $B_k$, $b_k$ and $Q_k$ are the dynamics weights, input weights, bias and covariance of discrete state $k$,
    * $H_k$, $D_k$, $d_k$ and $R_k$ are the emission weights, input weights, bias and covariance of discrete state $k$,
      shared by all states when ``single_subspace=True``.

    The parameters of the model are stored in a :class:`ParamsSLDS`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    :param num_states: number of discrete states
    :param state_dim: dimensionality of the continuous state
    :param emission_dim: dimensionality of the emissions
    :param input_dim: dimensionality of the inputs. Defaults to 0.
    :param single_subspace: whether all discrete states share one emission model. Defaults to True.
    :param dynamics_cov_type: ``"full"`` (default) or ``"diag"`` dynamics covariances
    :param emission_cov_type: ``"diag"`` (default) or ``"full"`` emission covariances
    :param initial_probs_concentration: concentration of the Dirichlet prior on the initial distribution
    :param transition_matrix_concentration: concentration of the Dirichlet prior on the rows of the transition matrix
    :param transition_matrix_stickiness: extra prior weight on the diagonal of the transition matrix
    :param dynamics_kwargs: optional hyperparameters of the dynamics component
    :param emission_kwargs: optional hyperparameters of the emission component
    :param driver_kwargs: optional fitting settings, see :class:`SLDS`
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
