
from dynamax.typed.types import *
from dynamax.utils import PSDToRealBijector


@dataclass
class MyLGSSM(SSM):
    state_dim: int
    emission_dim: int
    covariate_dim: int = 0
    has_dynamics_bias: bool = True
    has_emissions_bias: bool = True

    def initialize_params_old(self, key: PRNGKey) -> Tuple[Params, ParamProps]:
        m = jnp.zeros(self.state_dim)
        S = jnp.eye(self.state_dim)
        F = 0.99 * jnp.eye(self.state_dim)
        B = jnp.zeros((self.state_dim, self.covariate_dim))
        b = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        Q = 0.1 * jnp.eye(self.state_dim)
        H = jr.normal(key, (self.emission_dim, self.state_dim))
        D = jnp.zeros((self.emission_dim, self.covariate_dim))
        d = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        R = 0.1 * jnp.eye(self.emission_dim)
        params = dict(
            initial=dict(mean=m, cov=S),
            transitions=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )
        param_props = dict(
            initial=dict(mean=ParameterProperties(),
                         cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            transitions=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            emissions=dict(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        )
        return (params, param_props)

    def initialize_params(self, key: PRNGKey) -> Tuple[ParamsLGSSM, ParamPropsLGSSM]:
        m = jnp.zeros(self.state_dim)
        S = jnp.eye(self.state_dim)
        F = 0.99 * jnp.eye(self.state_dim)
        B = jnp.zeros((self.state_dim, self.covariate_dim))
        b = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        Q = 0.1 * jnp.eye(self.state_dim)
        H = jr.normal(key, (self.emission_dim, self.state_dim))
        D = jnp.zeros((self.emission_dim, self.covariate_dim))
        d = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        R = 0.1 * jnp.eye(self.emission_dim)
        params = ParamsLGSSM(
            initial_dist = GaussDist(mean=m, cov=S),
            dynamics = CondGaussDistLatent(weights=F, bias=b, input_weights=B, cov=Q),
            emissions = CondGaussDistObserved(weights=H, bias=d, input_weights=D, cov=R)
        )
        param_props = ParamPropsLGSSM(
            initial_dist = ParamPropsGauss(mean=ParameterProperties(),
                         cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            dynamics = ParamPropsCondGauss(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector))),
            emissions = ParamPropsCondGauss(weights=ParameterProperties(),
                          bias=ParameterProperties(),
                          input_weights=ParameterProperties(),
                          cov=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        )
        return (params, param_props)

    def filter(self, params: ParamsLGSSM, emissions: EmissionSeqVec, inputs: Optional[InputSeq]=None) -> GSSMPosterior:
        #print('filtering params ', params['initial']['mean'])
        print('filtering params ', params.initial_dist.mean)
        loglik = jnp.sum(emissions) # random 
        ntime = emissions.shape[0]
        return GSSMPosterior(marginal_loglik = loglik, filtered_means = jnp.zeros((ntime, self.state_dim)))

    def sample(self, params: ParamsLGSSM, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqVec, EmissionSeqVec]:
        states = jnp.zeros((num_timesteps, self.state_dim))
        emissions = jnp.zeros((num_timesteps, self.emission_dim))
        return (states, emissions)