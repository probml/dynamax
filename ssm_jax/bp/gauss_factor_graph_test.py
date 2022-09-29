from concurrent.futures.process import _MAX_WINDOWS_WORKERS
from functools import partial
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, jit

from ssm_jax.distributions import InverseWishart
from ssm_jax.bp.gauss_bp_utils import info_multiply, potential_from_conditional_linear_gaussian, pair_cpot_condition
from ssm_jax.bp.gauss_factor_graph import (GaussianVariableNode,
                                           CanonicalFactor,
                                           CanonicalPotential,
                                           GaussianFactorGraph,
                                           zeros_canonical_pot,
                                           update_all_messages,
                                           make_factor_graph,
                                           init_messages,
                                           calculate_all_beliefs,
                                           make_canonical_factor)

from ssm_jax.linear_gaussian_ssm.inference import lgssm_sample
from ssm_jax.linear_gaussian_ssm.info_inference import lgssm_info_smoother
from ssm_jax.linear_gaussian_ssm.info_inference_test import build_lgssm_moment_and_info_form

_all_close = lambda x,y: jnp.allclose(x,y,rtol=1e-3, atol=1e-3)

def canonical_factor_from_clg(A, u, Lambda, x, y, factorID):
    """Construct a CanoncialFactor from the parameters of a conditional linear Gaussian."""
    (Kxx, Kxy, Kyy), (hx, hy) = potential_from_conditional_linear_gaussian(A, u, Lambda)
    K = jnp.block([[Kxx, Kxy], [Kxy.T, Kyy]])
    h = jnp.concatenate((hx, hy))
    cpot = CanonicalPotential(eta=h, Lambda=K)
    return make_canonical_factor(factorID, [x, y], cpot)


def factor_graph_from_lgssm(lgssm_params, inputs, obs, T=None):
    """Unroll a linear gaussian state-space model into a factor graph."""
    if inputs is None:
        if T is not None:
            D_in = lgssm_params.dynamics_input_weights.shape[1]
            inputs = jnp.zeros((T, D_in))
        else:
            raise ValueError("One of `inputs` or `T` must not be None.")

    num_timesteps = len(inputs)
    Lambda0, mu0 = lgssm_params.initial_precision, lgssm_params.initial_mean
    latent_dim = len(mu0)

    latent_vars = [GaussianVariableNode(f"x{i}", latent_dim, zeros_canonical_pot(latent_dim))
                    for i in range(num_timesteps)]
    # Add informative prior to first time point.
    x0_prior = CanonicalPotential(eta=Lambda0 @ mu0, Lambda=Lambda0)
    latent_vars[0] = GaussianVariableNode("x0", latent_dim, x0_prior)

    B, b = lgssm_params.dynamics_input_weights, lgssm_params.dynamics_bias
    F, Q_prec = lgssm_params.dynamics_matrix, lgssm_params.dynamics_precision
    latent_net_inputs = vmap(jnp.dot, (None, 0))(B, inputs) + b
    latent_factors = [
        canonical_factor_from_clg(
            F, latent_net_inputs[i], Q_prec, latent_vars[i], latent_vars[i + 1], f"latent_{i},{i+1}"
        )
        for i in range(num_timesteps - 1)
    ]

    D, d = lgssm_params.emission_input_weights, lgssm_params.emission_bias
    H, R_prec = lgssm_params.emission_matrix, lgssm_params.emission_precision

    emission_net_inputs = vmap(jnp.dot, (None, 0))(D, inputs) + d
    emission_pots = vmap(potential_from_conditional_linear_gaussian, (None, 0, None))(
                         H, emission_net_inputs, R_prec)
    local_evidence_pot_chain = vmap(partial(pair_cpot_condition, obs_var=2))(emission_pots, obs)
    local_evidence_pots = [CanonicalPotential(eta, Lambda) for Lambda, eta in zip(*local_evidence_pot_chain)]

    def incorporate_local_evidence(var, pot):
        """Incorporate local evidence potential into Gaussian variable."""
        prior_plus_pot = info_multiply(var.prior, pot)
        return GaussianVariableNode(var.varID, var.dim, prior_plus_pot)

    latent_vars = [incorporate_local_evidence(var,pot) for var, pot in zip(latent_vars, local_evidence_pots)]

    fg = make_factor_graph(latent_vars, latent_factors)

    return fg

def test_gauss_factor_graph_lgssm():
    """Test that Gaussian chain belief propagation gets the same results as
     information form RTS smoother."""

    lgssm, lgssm_info = build_lgssm_moment_and_info_form()

    key = jr.PRNGKey(111)
    num_timesteps = 5 # Fewer timesteps so that we can run fewer iterations.
    input_size = lgssm.dynamics_input_weights.shape[1]
    inputs = jnp.zeros((num_timesteps, input_size))
    _, y = lgssm_sample(key, lgssm, num_timesteps, inputs=inputs)

    lgssm_info_posterior = lgssm_info_smoother(lgssm_info, y, inputs)

    fg = factor_graph_from_lgssm(lgssm_info,inputs, y)

    # Loopy bp.
    messages = init_messages(fg)
    for _ in range(num_timesteps):
        messages = update_all_messages(fg,messages)

    # Calculate final beliefs
    final_beliefs = calculate_all_beliefs(fg,messages)
    fg_etas = jnp.vstack([cpot.eta for cpot in final_beliefs.values()])
    fg_Lambdas = jnp.stack([cpot.Lambda for cpot in final_beliefs.values()])

    assert _all_close(fg_etas, lgssm_info_posterior.smoothed_etas)
    assert _all_close(fg_Lambdas, lgssm_info_posterior.smoothed_precisions)


def test_tree_factor_graph():

    key = jr.PRNGKey(0)
    dim = 2

    ### Construct variables in moment form ###
    IW = InverseWishart(dim, jnp.eye(dim)*0.1)
    key, subkey = jr.split(key)
    covs = jit(IW.sample,static_argnums=0)(5,subkey)

    key, subkey1  = jr.split(key)
    mu1 = jr.normal(subkey1,(dim,))
    Sigma1 = covs[0]

    key, subkey = jr.split(key)
    mu2 = jr.normal(subkey,(dim,))
    Sigma2 = covs[1]

    # x_3 | x_1, x_2 ~ N(x_3| A_31 x_1 + A_32 x_2, Sigma_{3|1,2})
    key, *subkeys = jr.split(key,3)
    A31 = jr.normal(subkeys[0],(dim,dim))
    A32 = jr.normal(subkeys[1],(dim,dim))
    Sigma3_cond = covs[2]
    mu3 = A31 @ mu1 + A32 @ mu2
    Sigma3 = Sigma3_cond + A31 @ Sigma1 @ A31.T + A32 @ Sigma2 @ A32.T

    # x_4 | x_3 ~ N(x_4 | A_4 x_3, Sigma_{3|4})
    key, subkey = jr.split(key)
    A4 = jr.normal(subkey,(dim,dim))
    Sigma4_cond = covs[3]
    mu4 = A4 @ mu3
    Sigma4 = Sigma4_cond + A4 @ Sigma3 @ A4.T

    # x_5 | x_3 ~ N(x_5 | A_5 x_3, Sigma_{5|4})
    key, subkey = jr.split(key)
    A5 = jr.normal(subkey,(dim,dim))
    Sigma5_cond = covs[4]
    mu5 = A5 @ mu3
    Sigma5 = Sigma5_cond + A5 @ Sigma3 @ A5.T

    ### Construct variables and factors in Canonical Form ###
    Lambda1 = jnp.linalg.inv(Sigma1)
    eta1 = Lambda1 @ mu1
    prior_x1 = CanonicalPotential(eta1, Lambda1)

    Lambda2 = jnp.linalg.inv(Sigma2)
    eta2 = Lambda2 @ mu2
    prior_x2 = CanonicalPotential(eta2, Lambda2)

    x1_var = GaussianVariableNode(1, dim, prior_x1)
    x2_var = GaussianVariableNode(2, dim, prior_x2)
    x3_var = GaussianVariableNode(3, dim, zeros_canonical_pot(dim))
    x4_var = GaussianVariableNode(4, dim, zeros_canonical_pot(dim))
    x5_var = GaussianVariableNode(5, dim, zeros_canonical_pot(dim))

    offset = jnp.zeros(dim)
    Lambda3_cond = jnp.linalg.inv(Sigma3_cond)
    A3_joint = jnp.hstack((A31,A32))
    (Kxx, Kxy, Kyy), (hx,hy) = potential_from_conditional_linear_gaussian(A3_joint,
                                                                        offset,
                                                                        Lambda3_cond)
    K = jnp.block([[Kxx, Kxy],
                    [Kxy.T, Kyy]])
    h = jnp.concatenate((hx,hy))
    cpot_123 = CanonicalPotential(eta=h, Lambda=K)
    factor_123 = make_canonical_factor("factor_123", [x1_var, x2_var, x3_var], cpot_123)

    Lambda4_cond = jnp.linalg.inv(Sigma4_cond)
    factor_34 = canonical_factor_from_clg(A4, offset, Lambda4_cond, x3_var, x4_var, "factor_34")

    Lambda5_cond = jnp.linalg.inv(Sigma5_cond)
    factor_35 = canonical_factor_from_clg(A5, offset, Lambda5_cond, x3_var, x5_var, "factor_35")

    # Build factor graph.
    var_nodes = [x1_var, x2_var, x3_var, x4_var, x5_var]
    factors = [factor_123, factor_34, factor_35]
    fg = make_factor_graph(var_nodes, factors)

    # Loopy BP
    messages = init_messages(fg)
    for _ in range(5):
        messages = update_all_messages(fg,messages)

    # Extract marginal etas and Lambas from factor graph.
    final_beliefs = calculate_all_beliefs(fg,messages)
    fg_etas = jnp.vstack([cpot.eta for cpot in final_beliefs.values()])
    fg_Lambdas = jnp.stack([cpot.Lambda for cpot in final_beliefs.values()])

    # Convert to moment form
    fg_means = vmap(jnp.linalg.solve)(fg_Lambdas, fg_etas)
    fg_covs = jnp.linalg.inv(fg_Lambdas)

    means = jnp.vstack([mu1,mu2,mu3,mu4,mu5])
    covs = jnp.stack([Sigma1,Sigma2,Sigma3,Sigma4,Sigma5])

    # Compare to moment form marginals.
    assert jnp.allclose(fg_means,means,rtol=1e-2,atol=1e-2)
    assert jnp.allclose(fg_covs,covs,rtol=1e-2,atol=1e-2)