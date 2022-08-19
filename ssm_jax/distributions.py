from typing import Any, Optional
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class InverseWishart(tfd.TransformedDistribution):

    def __init__(self, df, scale):
        """Implementation of an inverse Wishart distribution as a transformation of
        a Wishart distribution. This distribution is defined by a scalar degrees of
        freedom `df` and a scale matrix, `scale`.

        #### Mathematical Details
        The probability density function (pdf) is,
        ```none
        pdf(X; df, scale) = det(X)**(-0.5 (df+k+1)) exp(-0.5 tr[inv(X) scale]) / Z
        Z = 2**(0.5 df k) Gamma_k(0.5 df) |det(scale)|**(-0.5 df)
        ```

        where
        * `df >= k` denotes the degrees of freedom,
        * `scale` is a symmetric, positive definite, `k x k` matrix,
        * `Z` is the normalizing constant, and,
        * `Gamma_k` is the [multivariate Gamma function](
            https://en.wikipedia.org/wiki/Multivariate_gamma_function).

        Args:
            df (_type_): _description_
            scale (_type_): _description_
        """
        self._df = df
        self._scale = scale
        # Compute the Cholesky of the inverse scale to parameterize a
        # Wishart distribution
        dim = scale.shape[-1]
        eye = jnp.broadcast_to(jnp.eye(dim), scale.shape)
        cho_scale = jnp.linalg.cholesky(scale)
        inv_scale_tril = solve_triangular(cho_scale, eye, lower=True)

        super().__init__(
            tfd.WishartTriL(df, scale_tril=inv_scale_tril),
            tfb.Chain([tfb.CholeskyOuterProduct(),
                       tfb.CholeskyToInvCholesky(),
                       tfb.Invert(tfb.CholeskyOuterProduct())]))
        
        self._parameters = dict(df=df,
                                scale=scale)

    @classmethod
    def _parameter_properties(self, dtype, num_classes=None):
        return dict(
            # Annotations may optionally specify properties, such as `event_ndims`,
            # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
            # the `ParameterProperties` documentation for details.
            df=tfp.util.ParameterProperties(event_ndims=0),
            scale=tfp.util.ParameterProperties(event_ndims=2))

    @property
    def df(self):
        return self._df

    @property
    def scale(self):
        return self._scale

    def _mean(self):
        dim = self.scale.shape[-1]
        df = jnp.array(self.df)[..., None, None] # at least 2d on the right
        assert self.df > dim + 1, "Mean only exists if df > dim + 1"
        return self.scale / (df - dim - 1)

    def _mode(self):
        dim = self.scale.shape[-1]
        df = jnp.array(self.df)[..., None, None] # at least 2d on the right
        return self.scale / (df + dim + 1)

    def _variance(self):
        """Compute the marginal variance of each entry of the matrix.
        """
        def _single_variance(df, scale):
            assert scale.ndim == 2
            assert df.shape == scale.shape
            dim = scale.shape[-1]
            diag = jnp.diag(scale)
            rows = jnp.arange(dim)[:, None].repeat(3, axis=1)
            cols = jnp.arange(dim)[None, :].repeat(3, axis=0)
            numer = (df - dim + 1) * scale**2 + (df - dim - 1) * diag[rows] * diag[cols]
            denom  = (df - dim) * (df - dim - 1)**2 * (df - dim - 3)
            return numer / denom

        dfs, scales = jnp.broadcast_arrays(jnp.array(self.df)[..., None, None], self.scale)
        if scales.ndim == 2:
            return _single_variance(dfs, scales)
        else:
            return vmap(_single_variance)(dfs, scales)


class NormalInverseWishart(tfd.JointDistributionSequential):
    def __init__(self, loc, mean_concentration, df, scale):
        """
        A normal inverse Wishart (NIW) distribution with
        TODO: Finish this description
        Args:
            loc:            \mu_0 in math above
            mean_concentration: \kappa_0
            df:             \nu
            scale:          \Psi
        """
        # Store hyperparameters.
        # Note: these should really be private.
        self._loc = loc
        self._mean_concentration = mean_concentration
        self._df = df
        self._scale = scale

        super(NormalInverseWishart, self).__init__([
            InverseWishart(df, scale),
            lambda Sigma: tfd.MultivariateNormalFullCovariance(
                loc, Sigma / mean_concentration)]
        )
        
        self._parameters = dict(loc=loc,
                                mean_concentration=mean_concentration,
                                df=df,
                                scale=scale)

    @property
    def loc(self):
        return self._loc

    @property
    def mean_concentration(self):
        return self._mean_concentration

    @property
    def df(self):
        return self._df

    @property
    def scale(self):
        return self._scale

    def _mode(self):
        r"""Solve for the mode. Recall,
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
        covariance = jnp.einsum("...,...ij->...ij", 1 / (self._df + dim + 2), self._scale)
        return covariance, self._loc


class MatrixNormalPrecision(tfd.TransformedDistribution):
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
    def _parameter_properties(self, dtype, num_classes=None):
        return dict(
            # Annotations may optionally specify properties, such as `event_ndims`,
            # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
            # the `ParameterProperties` documentation for details.
            loc = tfp.util.ParameterProperties(event_ndims=2),
            row_covariance=tfp.util.ParameterProperties(event_ndims=2),
            col_precision=tfp.util.ParameterProperties(event_ndims=2))
    
    @property
    def loc(self):
        return self._loc
    
    @property
    def row_covariance(self):
        return self._row_cov
    
    @property
    def col_precision(self):
        return self._col_precision
        
    def _mode(self):
        return self._loc


class MatrixNormalInverseWishart(tfd.JointDistributionSequential):
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
        self._matrix_normal_shape = loc.shape
        self._loc = loc
        self._col_precision = col_precision
        self._df = df
        self._scale = scale
        super().__init__([InverseWishart(df, scale),
                          lambda Sigma: MatrixNormalPrecision(loc, Sigma, col_precision)])
        
        self._parameters = dict(loc=loc,
                                col_precision=col_precision,
                                df=df,
                                scale=scale)
    
    @property
    def loc(self):
        return self._loc
    
    @property
    def col_precision(self):
        return self._col_precision
    
    @property
    def df(self):
        return self._df
    
    @property
    def scale(self):
        return self._scale
    
    def _mode(self):
        num_row, num_col = self._matrix_normal_shape
        covariance = jnp.einsum("...,...ij->...ij", 
                               1 / (self._df + num_row + num_col + 1), self._scale)
        return covariance, self._loc


###############################################################################

def niw_posterior_update(niw_prior, sufficient_stats):
    """Update the NormalInverseWishart distribution using sufficient statistics
    
    Returns:
        posterior NIW distribution
    """
    # extract parameters of the prior distribution
    loc_pri, precision_pri, df_pri, scale_pri = niw_prior.parameters.values()
    
    # unpack the sufficient statistics
    Sx, SxxT, N = sufficient_stats
    
    # compute parameters of the posterior distribution
    loc_pos = (precision_pri*loc_pri + Sx) / (precision_pri + N)
    precision_pos = precision_pri + N
    df_pos = df_pri + N
    scale_pos = scale_pri + SxxT \
        + precision_pri*jnp.outer(loc_pri, loc_pri) - precision_pos*jnp.outer(loc_pos, loc_pos)
    
    return NormalInverseWishart(loc=loc_pos, 
                                mean_concentration=precision_pos, 
                                df=df_pos, 
                                scale=scale_pos)


def mniw_posterior_update(mniw_prior, sufficient_stats):
    """Update the MatrixNormalInverseWishart distribution using sufficient statistics   

    Returns:
        the posterior MNIW distribution
    """
    # extract parameters of the prior distribution
    M_pri, V_pri, nu_pri, Psi_pri = mniw_prior.parameters.values()
    
    # unpack the sufficient statistics
    SxxT, SxyT, SyyT, N = sufficient_stats
    
    # compute parameters of the posterior distribution
    Sxx = V_pri + SxxT
    Sxy = SxyT + V_pri @ M_pri.T
    Syy = SyyT + M_pri @ V_pri @ M_pri.T
    M_pos = jnp.linalg.solve(Sxx, Sxy).T
    V_pos = Sxx
    nu_pos = nu_pri + N
    Psi_pos = Psi_pri + Syy - M_pos @ Sxy
    
    return MatrixNormalInverseWishart(loc=M_pos, 
                                      col_precision=V_pos, 
                                      df=nu_pos, 
                                      scale=Psi_pos)
    

def iw_posterior_update(iw_prior, sufficient_stats):
    """Update the InverseWishart distribution using sufficient statistics
    
    Returns:
        posterior IW distribution
    """
    # extract parameters of the prior distribution
    df_pri, scale_pri = iw_prior.parameters.values()
    
    # unpack the sufficient statistics
    SxxT, N = sufficient_stats
    
    # compute parameters of the posterior distribution
    df_pos = df_pri + N
    scale_pos = scale_pri + SxxT
    
    return InverseWishart(df=df_pos, 
                          scale=scale_pos)


