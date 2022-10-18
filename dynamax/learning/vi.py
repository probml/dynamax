import jax.numpy as np

def fit_vi(self, key, sample_size, emissions, inputs=None, M=100):
        """
        ADVI approximate the posterior distribtuion p of unconstrained global parameters
        with factorized multivatriate normal distribution:
        q = \prod_{k=1}^{K} q_k(mu_k, sigma_k),
        where K is dimension of p.

        The hyper-parameters of q to be optimized over are (mu_k, log_sigma_k))_{k=1}^{K}.

        The trick of reparameterization is employed to reduce the variance of SGD,
        which is achieved by written KL(q || p) as expectation over standard normal distribution
        so a sample from q is obstained by
        s = z * exp(log_sigma_k) + mu_k,
        where z is a sample from the standard multivarate normal distribtion.

        This implementation of ADVI uses fixed samples from q during fitting, instead of
        updating samples from q in each iteration, as in SGD.
        So the second order fixed optimization algorithm L-BFGS is used.

        Args:
            sample_size (int): number of samples to be returned from the fitted approxiamtion q.
            M (int): number of fixed samples from q used in evaluation of ELBO.

        Returns:
            Samples from the approximate posterior q
        """
        key0, key1 = jr.split(key)
        model_unc_params, fixed_params = to_unconstrained(self.params, self.param_props)
        params_flat, params_structure = flatten(model_unc_params)
        vi_dim = len(params_flat)

        std_normal = MVNDiag(jnp.zeros(vi_dim), jnp.ones(vi_dim))
        std_samples = std_normal.sample(seed=key0, sample_shape=(M,))
        std_samples = vmap(unflatten, (None, 0))(params_structure, std_samples)

        @jit
        def unnorm_log_pos(unc_params):
            """Unnormalzied log posterior of global parameters."""

            params = from_unconstrained(unc_params, fixed_params, self.param_props)
            log_det_jac = log_det_jac_constrain(unc_params, fixed_params, self.param_props)
            log_pri = self.log_prior(params) + log_det_jac
            batch_lls = self.marginal_log_prob(emissions, inputs, params)
            lp = log_pri + batch_lls.sum()
            return lp

        @jit
        def _samp_elbo(vi_params, std_samp):
            """Evaluate ELBO at one sample from the approximate distribution q.
            """
            vi_means, vi_log_sigmas = vi_params
            # unc_params_flat = vi_means + std_samp * jnp.exp(vi_log_sigmas)
            # unc_params = unflatten(params_structure, unc_params_flat)
            # With reparameterization, entropy of q evaluated at z is
            # sum(hyper_log_sigma) plus some constant depending only on z.
            _params = tree_map(lambda x, log_sig: x * jnp.exp(log_sig), std_samp, vi_log_sigmas)
            unc_params = tree_map(lambda x, mu: x + mu, _params, vi_means)
            q_entropy = flatten(vi_log_sigmas)[0].sum()
            return q_entropy + unnorm_log_pos(unc_params)

        objective = lambda x: -vmap(_samp_elbo, (None, 0))(x, std_samples).mean()

        # Fit ADVI with LBFGS algorithm
        initial_vi_means = model_unc_params
        initial_vi_log_sigmas = unflatten(params_structure, jnp.zeros(vi_dim))
        initial_vi_params = (initial_vi_means, initial_vi_log_sigmas)
        lbfgs = LBFGS(maxiter=1000, fun=objective, tol=1e-3, stepsize=1e-3, jit=True)
        (vi_means, vi_log_sigmas), _info = lbfgs.run(initial_vi_params)

        # Sample from the learned approximate posterior q
        _samples = std_normal.sample(seed=key1, sample_shape=(sample_size,))
        _vi_means = flatten(vi_means)[0]
        _vi_log_sigmas = flatten(vi_log_sigmas)[0]
        vi_samples_flat = _vi_means + _samples * jnp.exp(_vi_log_sigmas)
        vi_unc_samples = vmap(unflatten, (None, 0))(params_structure, vi_samples_flat)
        vi_samples = vmap(from_unconstrained, (0, None, None))(
            vi_unc_samples, fixed_params, self.param_props)

        return vi_samples
