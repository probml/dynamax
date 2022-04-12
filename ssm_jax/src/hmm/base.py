"""
Common utilities and structures (data containers) used across the HMM module
"""

from typing import NamedTuple, Optional

import chex


class HMMParams(NamedTuple):
    # TODO: This could support PyTrees and not only arrays.
    # Parameters
    initial_distribution: chex.Array
    transition: chex.Array
    emission: chex.Array

    # Properties
    time_invariant: bool
    n_steps: Optional[int] = None

    """
    A container defining a Hidden Markov model in all generality.
    
    Attributes:
        initial_distribution: 
            An array corresponding to the initial distribution of the HMM
        transition: 
            An array corresponding to the transition matrix of the HMM. It is possibly
            batched along the first dimension.
        emission: 
            An array corresponding to the emission matrix of the HMM. It is possibly
            batched along the first dimension.
        time_invariant: 
            An optional flag stating if the transition AND emission matrices are the same across time
            steps, in which case these need to not be batched. Default is `False`.
        n_steps: 
            An optional integer stating the number of time steps in the HMM. It is necessary if 
            `time_invariant` is `True`.
    """

    @property
    def T(self):
        """Safe attribute to get the number of time steps"""
        if self.time_invariant:
            return self.n_steps

        chex.assert_equal(self.transition.shape[0], self.emission.shape[0])
        return self.transition.shape[0]

    @property
    def dx(self):
        """Safe attribute to get the dimension of the latent state"""
        chex.assert_equal(self.transition.shape[-2], self.initial_distribution.shape[-1])
        chex.assert_equal(self.transition.shape[-2], self.transition.shape[-1])
        chex.assert_equal(self.transition.shape[-2], self.transition.shape[-1])
        return self.transition.shape[-2]

    @property
    def dy(self):
        """Safe attribute to get the dimension of the latent state"""
        return self.emission.shape[-2]




