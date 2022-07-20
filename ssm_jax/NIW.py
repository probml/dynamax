"""
This implementation of Normal Inverse Wishart distribution is directly copied from the note book of Scott Linderman:
'Implementing a Normal Inverse Wishart Distribution in Tensorflow Probability'
https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart.ipynb
and 
https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart_(Part_2).ipynb
"""
import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt
from functools import partial

tfd = tfp.distributions
tfb = tfp.bijectors


class NormalInverseWishart(tfd.JointDistributionNamed):
    def __init__(self, loc, mean_precision, df, scale, **kwargs):
        """
        A normal inverse Wishart (NIW) distribution with

        Args:
            loc:            \mu_0 in math above
            mean_precision: \kappa_0 
            df:             \nu
            scale:          \Psi 

        Returns: 
            A tfp.JointDistribution object.
        """
        # Store hyperparameters. 
        self._loc = loc
        self._mean_precision = mean_precision
        self._df = df
        self._scale = scale
        
        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        # Note: this could be done more efficiently.
        self.wishart_scale_tril = np.linalg.cholesky(np.linalg.inv(scale))

        super(NormalInverseWishart, self).__init__(dict(
            Sigma=lambda: tfd.TransformedDistribution(
                tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                tfb.Chain([tfb.CholeskyOuterProduct(),                 
                        tfb.CholeskyToInvCholesky(),                
                        tfb.Invert(tfb.CholeskyOuterProduct())
                        ])),
            mu=lambda Sigma: tfd.MultivariateNormalFullCovariance(
                loc, Sigma / mean_precision)
        ))

        # Replace the default JointDistributionNamed parameters with the NIW ones
        # because the JointDistributionNamed parameters contain lambda functions,
        # which are not jittable.
        self._parameters = dict(
            loc=loc,
            mean_precision=mean_precision,
            df=df,
            scale=scale
        )

    # These functions compute the pseudo-observations implied by the NIW prior
    # and convert sufficient statistics to a NIW posterior. We'll describe them
    # in more detail below.
    @property
    def natural_parameters(self):
        """Compute pseudo-observations from standard NIW parameters."""
        dim = self._loc.shape[-1]
        chi_1 = self._df + dim + 2
        chi_2 = np.einsum('...,...i->...i', self._mean_precision, self._loc)
        chi_3 = self._scale + self._mean_precision * \
            np.einsum("...i,...j->...ij", self._loc, self._loc)
        chi_4 = self._mean_precision
        return chi_1, chi_2, chi_3, chi_4

    @classmethod
    def from_natural_parameters(cls, natural_params):
        """Convert natural parameters into standard parameters and construct."""
        chi_1, chi_2, chi_3, chi_4 = natural_params
        dim = chi_2.shape[-1]
        df = chi_1 - dim - 2
        mean_precision = chi_4
        loc = np.einsum('..., ...i->...i', 1 / mean_precision, chi_2)
        scale = chi_3 - mean_precision * np.einsum('...i,...j->...ij', loc, loc)
        return cls(loc, mean_precision, df, scale)

    def _mode(self):
        """Solve for the mode. Recall,
        .. math::
            p(\mu, \Sigma) \propto
                \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)
        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        .. math::
            p(\mu^*, \Sigma) \propto IW(\Sigma | \nu_0 + 1, \Psi_0)
        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + d + 2)
        """
        dim = self._loc.shape[-1]
        covariance = np.einsum("...,...ij->...ij", 
                               1 / (self._df + dim + 2), self._scale)
        return self._loc, covariance

class InverseWishart(tfd.JointDistributionNamed):
    def __init__(self, df, scale, **kwargs):
        """
        An inverse Wishart (IW) distribution with

        Args: 
            df:             \nu
            scale:          \Psi 

        Returns: 
            A tfp.JointDistribution object.
        """
        # Store hyperparameters. 
        self._df = df
        self._scale = scale
        
        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        # Note: this could be done more efficiently.
        self.wishart_scale_tril = np.linalg.cholesky(np.linalg.inv(scale))

        super(InverseWishart, self).__init__(Sigma=lambda: tfd.TransformedDistribution(
                                             tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                                             tfb.Chain([tfb.CholeskyOuterProduct(),                 
                                             tfb.CholeskyToInvCholesky(),                
                                             tfb.Invert(tfb.CholeskyOuterProduct())]))
                                            )
        # Replace the default JointDistributionNamed parameters with the IW ones
        # because the JointDistributionNamed parameters contain lambda functions,
        # which are not jittable.
        self._parameters = dict(df=df, scale=scale)