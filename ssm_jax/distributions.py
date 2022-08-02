import jax.numpy as jnp
from jax import vmap
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
        inv_scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(scale))

        super().__init__(
            tfd.WishartTriL(df, scale_tril=inv_scale_tril),
            tfb.Chain([tfb.CholeskyOuterProduct(),
                       tfb.CholeskyToInvCholesky(),
                       tfb.Invert(tfb.CholeskyOuterProduct())]))

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


# class NormalInverseWishart(tfd.JointDistributionNamed):
#     def __init__(self, loc, mean_concentration, df, scale, **kwargs):
#         """
#         A normal inverse Wishart (NIW) distribution with
#         TODO: Finish this description
#         Args:
#             loc:            \mu_0 in math above
#             mean_concentration: \kappa_0
#             df:             \nu
#             scale:          \Psi
#         """
#         # Store hyperparameters.
#         # Note: these should really be private.
#         self._loc = loc
#         self._mean_concentration = mean_concentration
#         self._df = df
#         self._scale = scale

#         super(NormalInverseWishart, self).__init__(dict(
#             Sigma=lambda: InverseWishart(df, scale),
#             mu=lambda Sigma: tfd.MultivariateNormalFullCovariance(
#                 loc, Sigma / mean_concentration)
#         ))

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
