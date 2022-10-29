
from dynamax.typed.types import *

class MyLGSSM(SSM):
    def __init__(self,
                 state_dim: int,
                 emission_dim: int,
                 covariate_dim: int = 0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.covariate_dim = covariate_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

    def initialize_params(self, key: PRNGKey) -> Tuple[Params, ParamProps]:
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
            dynamics=dict(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=dict(weights=H, bias=d, input_weights=D, cov=R)
        )
        param_props = dict(
            initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
        )
        return (params, param_props)


    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> GSSMPosterior:
        print('filtering params ', params['initial']['mean'])
        loglik = jnp.sum(emissions)
        ntime = emissions.shape[0]
        return GSSMPosterior(marginal_loglik = loglik, filtered_means = jnp.zeros((ntime, self.state_dim)))

    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqVec, EmissionSeq]:
        states = jnp.zeros((num_timesteps, self.state_dim))
        emissions = jnp.zeros((num_timesteps, self.emission_dim))
        return (states, emissions)



