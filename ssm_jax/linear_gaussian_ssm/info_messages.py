import jax.numpy as np
from jax import lax, value_and_grad
from jax.scipy.linalg import solve_triangular


def block_tridiag_mvn_log_normalizer(precision_diag_blocks, precision_lower_diag_blocks, linear_potential):
    """
    Compute the log normalizing constant for a multivariate normal distribution
    with natural parameters :math:`J` and :math:`h` with density,
    ..math:
        \log p(x) = -1/2 x^\top J x + h^\top x - \log Z

    where the log normalizer is
    ..math:
        \log Z = N/2 \log 2 \pi - \log |J| + 1/2 h^T J^{-1} h

    and :math:`N` is the dimensionality.

    Typically, computing the log normalizer is cubic in N, but if :math:`J` is
    block tridiagonal, it can be computed in O(N) time. Specifically, suppose
    J is TDxTD with blocks of size D on the diagonal and first off diagonal.
    Since J is symmetric, we can represent the matrix by only its diagonal and
    first lower diagonal blocks. This is exactly the type of precision matrix
    we encounter with linear Gaussian dynamical systems. This code computes its
    log normalizer using the so-called "information form Kalman filter."

    Args:

    precision_diag_blocks:          Shape (T, D, D) array of the diagonal blocks
                                    of a shape (TD, TD) precision matrix.
    precision_lower_diag_blocks:    Shape (T-1, D, D) array of the lower diagonal
                                    blocks of a shape (TD, TD) precision matrix.
    linear_potential:               Shape (T, D) array of linear potentials of a
                                    TD dimensional multivariate normal distribution
                                    in information form.

    Returns:

    log_normalizer:                 The scalar log normalizing constant.
    (filtered_Js, filtered_hs):     The precision and linear potentials of the
                                    Gaussian filtering distributions in information
                                    form, with shape (T, D, D) and (T, D) respectively.
    """
    # Shorthand names
    J_diag = precision_diag_blocks
    J_lower_diag = precision_lower_diag_blocks
    h = linear_potential

    # extract dimensions
    num_timesteps, dim = J_diag.shape[:2]

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = np.concatenate((J_lower_diag, np.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        # Predict
        sqrt_Jc = np.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * np.log(2 * np.pi)
        log_Z += -np.sum(np.log(np.diag(sqrt_Jc)))  # sum these terms only to get approx log|J|
        log_Z += 0.5 * np.dot(trm1.T, trm1)
        Jp = -np.dot(trm2.T, trm2)
        hp = -np.dot(trm2.T, trm1)

        # Alternative predict step:
        # log_Z = 0.5 * dim * np.log(2 * np.pi)
        # log_Z += -0.5 * np.linalg.slogdet(Jc)[1]
        # log_Z += 0.5 * np.dot(hc, np.linalg.solve(Jc, hc))
        # Jp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, J_lower_diag_pad[t].T))
        # hp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, hc))

        new_carry = Jp, hp, lp + log_Z
        return new_carry, (Jc, hc)

    # Initialize
    Jp0 = np.zeros((dim, dim))
    hp0 = np.zeros((dim,))
    (_, _, log_Z), (filtered_Js, filtered_hs) = lax.scan(marginalize, (Jp0, hp0, 0), np.arange(num_timesteps))
    return log_Z, (filtered_Js, filtered_hs)


def block_tridiag_mvn_expectations(precision_diag_blocks, precision_lower_diag_blocks, linear_potential):
    # Run message passing code to get the log normalizer, the filtering potentials,
    # and the expected values of x. Technically, the natural parameters are -1/2 J
    # so we need to do a little correction of the gradients to get the expectations.
    f = value_and_grad(block_tridiag_mvn_log_normalizer, argnums=(0, 1, 2), has_aux=True)
    (log_normalizer, _), grads = f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential)

    # Correct for the -1/2 J -> J implementation
    ExxT = -2 * grads[0]
    ExxnT = -grads[1]
    Ex = grads[2]
    return log_normalizer, Ex, ExxT, ExxnT


def lds_to_block_tridiag(lds, data, inputs):
    # Shorthand names for parameters
    m0 = lds.initial_mean
    Q0 = lds.initial_covariance
    A = lds.dynamics_matrix
    B = lds.dynamics_input_weights
    Q = lds.dynamics_noise_covariance
    C = lds.emissions_matrix
    D = lds.emissions_input_weights
    R = lds.emissions_noise_covariance
    T = len(data)

    # diagonal blocks of precision matrix
    J_diag = np.array([np.dot(C(t).T, np.linalg.solve(R(t), C(t))) for t in range(T)])
    J_diag = J_diag.at[0].add(np.linalg.inv(Q0))
    J_diag = J_diag.at[:-1].add(np.array([np.dot(A(t).T, np.linalg.solve(Q(t), A(t))) for t in range(T - 1)]))
    J_diag = J_diag.at[1:].add(np.array([np.linalg.inv(Q(t)) for t in range(0, T - 1)]))

    # lower diagonal blocks of precision matrix
    J_lower_diag = np.array([-np.linalg.solve(Q(t), A(t)) for t in range(T - 1)])

    # linear potential
    h = np.array([np.dot(data[t] - D(t) @ inputs[t], np.linalg.solve(R(t), C(t))) for t in range(T)])
    h = h.at[0].add(np.linalg.solve(Q0, m0))
    h = h.at[:-1].add(np.array([-np.dot(A(t).T, np.linalg.solve(Q(t), B(t) @ inputs[t])) for t in range(T - 1)]))
    h = h.at[1:].add(np.array([np.linalg.solve(Q(t), B(t) @ inputs[t]) for t in range(T - 1)]))

    return J_diag, J_lower_diag, h
