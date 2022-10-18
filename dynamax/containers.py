import chex

@chex.dataclass
class GSSMPosterior:
    """Simple wrapper for properties of an GSSM posterior distribution.

    Attributes:
            marginal_loglik: marginal log likelihood of the data
            filtered_means: (T,D_hid) array,
                E[x_t | y_{1:t}, u_{1:t}].
            filtered_covariances: (T,D_hid,D_hid) array,
                Cov[x_t | y_{1:t}, u_{1:t}].
            smoothed_means: (T,D_hid) array,
                E[x_t | y_{1:T}, u_{1:T}].
            smoothed_covariances: (T,D_hid,D_hid) array of smoothed marginal covariances,
                Cov[x_t | y_{1:T}, u_{1:T}].
            smoothed_cross: (T-1, D_hid, D_hid) array of smoothed cross products,
                E[x_t x_{t+1}^T | y_{1:T}, u_{1:T}].
    """

    marginal_loglik: chex.Scalar = None
    filtered_means: chex.Array = None
    filtered_covariances: chex.Array = None
    smoothed_means: chex.Array = None
    smoothed_covariances: chex.Array = None
    smoothed_cross_covariances: chex.Array = None
