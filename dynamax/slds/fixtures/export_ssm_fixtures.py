"""
Export reference values from ``lindermanlab/ssm`` (commit eb6c8aa) for ``dynamax/slds/ssm_oracle_test.py``
and ``dynamax/slds/benchmark_test.py``. Runs in an environment with ``ssm`` installed (see README.md);
dynamax is not needed.

    python export_ssm_fixtures.py                 # ssm_fixtures.npz: deterministic sub-steps
    python export_ssm_fixtures.py --benchmark     # ssm_benchmark.npz: recovery over seeds (slow)
"""
import argparse
import copy
import os

import numpy as np
import numpy.random as npr

import ssm
import ssm.variational as variational
from ssm.util import find_permutation

HERE = os.path.dirname(os.path.abspath(__file__))
K, D, N, T = 3, 2, 8, 100   # N > D * K: ssm's multi-subspace PCA needs D * K components


class SampleRecorder:
    """Wrap ``ssm.variational.kalman_info_sample`` to record every q(x) sample and the N(0, I) noise that made it.

    Inside the ``with`` block ssm's sampler is swapped for ``wrapped``: it snapshots numpy's RNG state, lets the
    real sampler run, rewinds to the snapshot, draws the same normals itself, then restores the RNG to where ssm
    left it so nothing downstream changes.
    """

    def __init__(self):
        self.samples = []
        self.original = variational.kalman_info_sample

    def __enter__(self):
        recorder = self

        def wrapped(*args, **kwargs):
            state = npr.get_state()
            sample = recorder.original(*args, **kwargs)
            after = npr.get_state()
            npr.set_state(state)
            noise = npr.randn(*sample.shape)      # the single draw inside _kalman_info_sample
            npr.set_state(after)
            recorder.samples.append((np.array(sample), noise))
            return sample

        variational.kalman_info_sample = wrapped
        return self

    def __exit__(self, *exc):
        variational.kalman_info_sample = self.original


def transition_arrays(transitions):
    """ssm transition parameters as plain arrays, by class."""
    name = type(transitions).__name__
    if name == "StationaryTransitions":
        return dict(transition_matrix=transitions.transition_matrix)
    if name == "RecurrentTransitions":
        return dict(log_Ps=transitions.log_Ps, Rs=transitions.Rs, Ws=transitions.Ws)
    if name == "RecurrentOnlyTransitions":
        return dict(r=transitions.r, Rs=transitions.Rs, Ws=transitions.Ws)
    raise ValueError(name)


def export_case(name, transitions, single_subspace, seed, out):
    """One configuration: parameters, data, one discrete update, one continuous update, and the M-steps."""
    npr.seed(seed)
    slds = ssm.SLDS(N, K, D, transitions=transitions, dynamics="gaussian", emissions="gaussian", single_subspace=single_subspace)
    z, x, y = slds.sample(T)
    datas, inputs, masks, tags = [y], [np.zeros((T, 0))], [np.ones_like(y, dtype=bool)], [None]
    slds.initialize(datas, inputs, masks, tags, num_init_iters=10)

    put = lambda key, value: out.__setitem__(f"{name}/{key}", np.asarray(value))
    put("z", z); put("x", x); put("y", y)
    put("pi0", slds.init_state_distn.initial_state_distn)
    for key, value in transition_arrays(slds.transitions).items():
        put(f"trans/{key}", value)
    dyn = slds.dynamics
    put("dyn/As", dyn.As); put("dyn/bs", dyn.bs); put("dyn/Vs", dyn.Vs); put("dyn/Sigmas", dyn.Sigmas)
    put("dyn/mu_init", dyn.mu_init); put("dyn/Sigmas_init", dyn.Sigmas_init)
    emis = slds.emissions
    put("emis/Cs", emis.Cs); put("emis/ds", emis.ds); put("emis/Fs", emis.Fs); put("emis/inv_etas", emis.inv_etas)

    # Variational posterior: the initial q(x) is N(invert(y), 1e-2 I). One discrete update (one x sample), then
    # one Newton update of q(x) from the initial mean.
    with SampleRecorder() as rec:
        q = variational.SLDSStructuredMeanFieldVariationalPosterior(slds, datas, inputs, masks, tags)
        x0 = q.mean_continuous_states[0]
        put("x0", x0)
        slds._fit_laplace_em_discrete_state_update(q, datas, inputs, masks, tags, num_samples=1)
    x_sample, noise = rec.samples[0]
    put("x_sample", x_sample); put("x_sample_noise", noise)
    params0 = q.continuous_state_params[0]
    for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs", "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        put(f"q0/{key}", params0[key])
    dparams = q.discrete_state_params[0]
    Ez, Ezzp1, log_Z = q.discrete_expectations[0]
    # ssm keeps stationary transitions as one (1, K, K) matrix with time-summed pairwise expectations; dynamax takes (K, K)
    squeeze = (lambda a: a[0]) if transitions == "standard" else (lambda a: a)
    put("qz/pi0", dparams["pi0"]); put("qz/Ps", squeeze(dparams["Ps"])); put("qz/log_likes", dparams["log_likes"])
    put("qz/Ez", Ez); put("qz/Ezzp1", squeeze(Ezzp1)); put("qz/log_Z", log_Z)

    # Hessian blocks, h = J x and the objective at a fixed point (the x sample).
    J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = slds._laplace_neg_hessian_params(y, inputs[0], masks[0], None, x_sample, Ez, Ezzp1)
    h_ini, h_dyn_1, h_dyn_2, h_obs = slds._laplace_neg_hessian_params_to_hs(x_sample, J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs)
    for key, value in dict(J_ini=J_ini, J_dyn_11=J_dyn_11, J_dyn_21=J_dyn_21, J_dyn_22=J_dyn_22, J_obs=J_obs,
                           h_ini=h_ini, h_dyn_1=h_dyn_1, h_dyn_2=h_dyn_2, h_obs=h_obs).items():
        put(f"hess/{key}", value)
    put("hess/neg_expected_log_joint", slds._laplace_neg_expected_log_joint(y, inputs[0], masks[0], None, x_sample, Ez, Ezzp1, scale=1))

    slds._fit_laplace_em_continuous_state_update(q, datas, inputs, masks, tags, continuous_optimizer="newton",
                                                 continuous_tolerance=1e-10, continuous_maxiter=200)
    put("x_star", q.mean_continuous_states[0])
    params1 = q.continuous_state_params[0]
    for key in ["J_ini", "J_dyn_11", "J_dyn_21", "J_dyn_22", "J_obs", "h_ini", "h_dyn_1", "h_dyn_2", "h_obs"]:
        put(f"q1/{key}", params1[key])
    log_Z, Ex, Sigmas, ExxnT = q.continuous_expectations[0]      # kalman_info_smoother outputs of the updated q(x)
    put("q1/log_Z", log_Z); put("q1/Ex", Ex); put("q1/Sigmas", Sigmas); put("q1/ExxnT", ExxnT)
    put("q1/continuous_entropy", q._continuous_entropy())     # ssm's discrete part is biased by its +1e-16 log guard

    # M-steps on copies: dynamics with the exact continuous expectations, emissions and transitions with the x sample.
    xmasks = [np.ones_like(x_sample, dtype=bool)]
    dyn = copy.deepcopy(slds.dynamics)
    dyn.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags, continuous_expectations=q.continuous_expectations)
    put("mstep/As", dyn.As); put("mstep/bs", dyn.bs); put("mstep/Vs", dyn.Vs); put("mstep/Sigmas", dyn.Sigmas)
    emis = copy.deepcopy(slds.emissions)
    emis.m_step(q.discrete_expectations, [x_sample], datas, inputs, masks, tags)
    put("mstep/Cs", emis.Cs); put("mstep/ds", emis.ds); put("mstep/Fs", emis.Fs); put("mstep/inv_etas", emis.inv_etas)
    if transitions != "standard":
        trans = copy.deepcopy(slds.transitions)
        trans.m_step(q.discrete_expectations, [x_sample], inputs, xmasks, tags, num_iters=1000)
        for key, value in transition_arrays(trans).items():
            put(f"mstep/trans/{key}", value)
        # the expected log transition term alone (ssm's Dirichlet log prior omits the normalizing constant)
        put("mstep/trans/expected_log_trans", np.sum(Ezzp1 * trans.log_transition_matrices(x_sample, inputs[0], xmasks[0], None)))
        put("mstep/trans/log_prior", trans.log_prior())
    print(f"{name}: exported ({len(out)} arrays so far)")


def export_fixtures(path):
    out = {}
    seed = 0
    for transitions in ["standard", "recurrent", "recurrent_only"]:
        for single_subspace in [True, False]:
            export_case(f"{transitions}/{'single' if single_subspace else 'multi'}", transitions, single_subspace, seed, out)
            seed += 1
    np.savez(path, **out)
    print(f"wrote {path}")


def export_benchmark(path, num_seeds, num_timesteps=1000, num_iters=50):
    """ssm's recovery on the tutorial problem (K=3, D=2, N=10): state accuracy and aligned latent RMSE per seed."""
    Kb, Db, Nb = 3, 2, 10
    rotation = lambda a: 0.95 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = {}
    for seed in range(num_seeds):
        npr.seed(1000 + seed)
        true = ssm.SLDS(Nb, Kb, Db, transitions="standard", dynamics="gaussian", emissions="gaussian", single_subspace=True)
        true.init_state_distn.log_pi0 = np.log(np.ones(Kb) / Kb)
        true.transitions.log_Ps = np.log(0.9 * np.eye(Kb) + 0.1 / Kb)
        true.dynamics.As = np.stack([rotation((k + 1) * np.pi / 20) for k in range(Kb)])
        true.dynamics.bs = np.array([[1.0, 0.0], [-0.5, 0.8], [-0.5, -0.8]])
        true.dynamics.Vs = np.zeros((Kb, Db, 0))
        true.dynamics.Sigmas = np.tile(0.1 * np.eye(Db), (Kb, 1, 1))
        true.dynamics.mu_init = np.zeros((Kb, Db)); true.dynamics.Sigmas_init = np.tile(np.eye(Db), (Kb, 1, 1))
        true.emissions.Cs = npr.randn(1, Nb, Db); true.emissions.ds = np.zeros((1, Nb)); true.emissions.Fs = np.zeros((1, Nb, 0))
        true.emissions.inv_etas = np.log(0.1) * np.ones((1, Nb))
        z, x, y = true.sample(num_timesteps)

        model = ssm.SLDS(Nb, Kb, Db, transitions="standard", dynamics="gaussian", emissions="gaussian", single_subspace=True)
        elbos, q = model.fit([y], method="laplace_em", variational_posterior="structured_meanfield", num_iters=num_iters,
                             initialize=True, num_init_iters=50, verbose=0)
        x_hat = q.mean_continuous_states[0]
        z_hat = model.most_likely_states(x_hat, y)
        perm = find_permutation(z_hat, z)
        accuracy = np.mean(perm[z_hat] == z)
        design = np.column_stack([x_hat, np.ones(num_timesteps)])
        aligned = design @ np.linalg.lstsq(design, x, rcond=None)[0]
        rmse = np.sqrt(np.mean((aligned - x) ** 2))
        out[f"seed{seed}/y"] = y; out[f"seed{seed}/z"] = z; out[f"seed{seed}/x"] = x
        out[f"seed{seed}/accuracy"] = accuracy; out[f"seed{seed}/rmse"] = rmse; out[f"seed{seed}/elbos"] = np.asarray(elbos)
        print(f"seed {seed}: accuracy {accuracy:.3f}, rmse {rmse:.3f}, final elbo {elbos[-1]:.1f}")
    out["num_seeds"] = num_seeds
    np.savez(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="export the recovery benchmark instead of the sub-steps")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.benchmark:
        export_benchmark(args.out or os.path.join(HERE, "ssm_benchmark.npz"), args.seeds)
    else:
        export_fixtures(args.out or os.path.join(HERE, "ssm_fixtures.npz"))
