"""
This implementation of Normal Inverse Wishart distribution is directly copied from the note book of Scott Linderman:
'Implementing a Normal Inverse Wishart Distribution in Tensorflow Probability'
https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart.ipynb
and 
https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart_(Part_2).ipynb
"""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from typing import Any, Optional

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
        self.wishart_scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(scale))

        super().__init__(dict(Sigma=tfd.TransformedDistribution(
                                tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                                tfb.Chain([tfb.CholeskyOuterProduct(),                 
                                           tfb.CholeskyToInvCholesky(),                
                                           tfb.Invert(tfb.CholeskyOuterProduct())])),
                              mu=lambda Sigma: tfd.MultivariateNormalFullCovariance(
                                loc, Sigma / mean_precision)))

        # Replace the default JointDistributionNamed parameters with the NIW ones
        # because the JointDistributionNamed parameters contain lambda functions,
        # which are not jittable.
        self._parameters = dict(loc=loc,
                                mean_precision=mean_precision,
                                df=df,
                                scale=scale)

    @property
    def mode(self):
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
        covariance = jnp.einsum("...,...ij->...ij", 
                               1 / (self._df + dim + 2), self._scale)
        return self._loc, covariance
    

class InverseWishart(tfd.TransformedDistribution):
    def __init__(self, df, scale):
        """
        An inverse Wishart (IW) distribution with

        Args: 
            df:    \nu
            scale: \Psi 

        Returns: 
            A tfp.Distribution object.
        """
        # Store hyperparameters. 
        self._df = df
        self._scale = scale
        
        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        self.wishart_scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(scale))

        super().__init__(tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                         tfb.Chain([tfb.CholeskyOuterProduct(),                 
                                    tfb.CholeskyToInvCholesky(),                
                                    tfb.Invert(tfb.CholeskyOuterProduct())]))
        
        # Replace the default TransformedDistribution parameters with the IW ones
        self._parameters = dict(df=df, scale=scale)
    
    # Override the _parameter_properties method of tfd.TransformedDistribution
    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype,
                                                      num_classes=num_classes)
        del td_properties['bijector']
        return td_properties
    
    @property
    def mode(self):
        dim = self._scale.shape[-1]
        covariance = jnp.einsum("...,...ij->...ij", 
                                1 / (self._df + dim + 1), self._scale)
        return covariance        
    
    
class MatrixNormal(tfd.TransformedDistribution):
    def __init__(self, loc, row_covariance, col_precision):
        """A matrix normal distribution

        Args:
            loc:            mean value of the matrix
            row_covariance: covariance matrix of rows of the matrix 
            col_precision:  precision matrix (inverse of covariance) of columns of the matrix
            
        Returns: 
            A tfp.Distribution object.
        """
        self._shape = loc.shape
        self._loc = loc
        self._row_cov = row_covariance
        self._col_precision = col_precision
        self._col_cov = jnp.linalg.inv(col_precision)
        # Vectorize by row, which is consistent with the tfb.Reshape bijector
        self._vec_mean = jnp.ravel(loc) 
        self._vec_cov = jnp.kron(row_covariance, self._col_cov)
        super().__init__(tfd.MultivariateNormalFullCovariance(self._vec_mean, self._vec_cov),
                         tfb.Reshape(event_shape_out=self._shape))
        
        # Replace the default MultivariateNormalFullCovariance parameters with the MatrixNormal ones
        self._parameters = dict(loc=loc, 
                                row_covariance=row_covariance,
                                col_precision=col_precision)
        
    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype,
                                                      num_classes=num_classes)
        del td_properties['bijector']
        return td_properties
    
    @property
    def row_covariance(self):
        return self._row_cov
    
    @property
    def col_precision(self):
        return self._col_precision
        
    @property
    def mode(self):
        return self._loc
    
    
class MatrixNormalInverseWishart(tfd.JointDistributionNamed):
    def __init__(self, loc, col_precision, df, scale):
        """A matrix normal inverse Wishart (MNIW) distribution

        Args:
            loc:           mean value matrix of the matrix normal distribution
            col_precision: column precision matrix (the inverse of the column covariance)
                           of the matrix normal ditribution
            df:            degree of freedom parameter of the inverse Wishart distribution
            scale:         the scale matrix of the inverse Wishart distribution
        
        Returns: 
            A tfp.JointDistribution object.
        """
        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        self.wishart_scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(scale))
        self._matrix_normal_shape = loc.shape
        self._loc = loc
        self._col_precision = col_precision
        self._col_cov = jnp.linalg.inv(col_precision)
        # Vectorize by row, which is consistent with the tfb.Reshape bijector
        self._vec_mean = jnp.ravel(loc)
        self._df = df
        self._scale = scale
        super().__init__(dict(
            Sigma=tfd.TransformedDistribution(
                    tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                    tfb.Chain([tfb.CholeskyOuterProduct(),                 
                               tfb.CholeskyToInvCholesky(),                
                               tfb.Invert(tfb.CholeskyOuterProduct())])),
            Matrix=lambda Sigma: tfd.TransformedDistribution(
                    tfd.MultivariateNormalFullCovariance(self._vec_mean, 
                                                         jnp.kron(Sigma, self._col_cov)),
                    tfb.Reshape(event_shape_out=self._matrix_normal_shape))
            ))
        self._parameters = dict(loc=loc,
                                col_precision=col_precision,
                                df=df,
                                scale=scale)
    
    @property
    def mode(self):
        num_row, num_col = self._matrix_normal_shape
        covariance = jnp.einsum("...,...ij->...ij", 
                               1 / (self._df + num_row + num_col + 1), self._scale)
        return self._loc, covariance
    