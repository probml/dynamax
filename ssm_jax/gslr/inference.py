import jax.numpy as jnp

def statistical_linear_regression(mu, Sigma, m, S, C):
    """Return moment-matching affine coefficients and approximation noise variance
    given prior and true posterior moments.
        g(x) \approx Ax + b + e where e ~ N(0, Omega)
        p(x) = N(x | mu, Sigma)
        m = E[g(x)]
        S = Var[g(x)]
        C = Cov[x, g(x)]

    Args:
        mu (D_hid): prior mean.
        Sigma (D_hid, D_hid): prior covariance.
        m (D_obs): E[g(x)].
        S (D_obs, D_obs): Var[g(x)]
        C (D_hid, D_obs): Cov[x, g(x)]

    Returns:
        A (D_obs, D_hid): _description_
        b (D_obs):
        Omega (D_obs, D_obs): 
    """    
    A = jnp.linalg.solve(Sigma.T, C).T
    b = m - A @ mu
    Omega = S - A @ Sigma @ A.T
    return A, b, Omega