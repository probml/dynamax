
from dynamax.typed.types import *

class MyHMM(SSM):
    def __init__(self,
                 nstates: int,
                 emission_dim: int,
                 covariate_dim: int = 0):
        self.nstates = nstates
        self.emission_dim = emission_dim
        self.covariate_dim = covariate_dim

    def initialize_params(self, rng_key: PRNGKey) -> Tuple[Params, ParamProps]:
        del rng_key
        Nz, Ny = self.nstates, self.emission_dim
        params = dict(
            initial=dict(probs=jnp.ones(Nz) / (Nz*1.0)),
            transitions=dict(transition_matrix=0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz),
            emissions=dict(means=jnp.zeros((Nz, Ny)), scales=jnp.ones((Nz, Ny)))
        )
        param_props = dict(
            initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions=dict(means=ParameterProperties(), scales=ParameterProperties(constrainer=tfb.Softplus(), trainable=False))
        )
        return (params, param_props)


    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        print('filtering params ', params['initial']['probs'])
        loglik = jnp.sum(emissions)
        ntime = emissions.shape[0]
        return HMMPosterior(marginal_loglik = loglik, filtered_probs = jnp.zeros((ntime, self.nstates)))

    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeq]:
        states = jnp.zeros(num_timesteps, dtype=int)
        emissions = jnp.zeros((num_timesteps, self.emission_dim))
        return (states, emissions)