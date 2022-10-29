
from dynamax.typed.types import *
from dynamax.utils import PSDToRealBijector

@dataclass
class HMM(SSM):
    num_states: int
    emission_dim: int
    covariate_dim: int = 0

    def filter(self, params: Params, emissions: EmissionSeq, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        pass

    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeq]:
        pass

@dataclass
class HMMDiscreteOutput(HMM):
    def filter(self, params: Params, emissions: EmissionSeqDiscrete, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        pass

    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeqDiscrete]:
        pass

@dataclass
class HMMVecOutput(HMM):
    def filter(self, params: Params, emissions: EmissionSeqVec, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        pass

    def sample(self, params: Params, rng_key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeqVec]:
        pass

@dataclass
class MyHMMD(HMMDiscreteOutput):

    def initialize_params_old(self, key: PRNGKey) -> Tuple[Params, ParamProps]:
        Nz, Ny = self.num_states, self.emission_dim
        params = dict(
            initial=dict(probs=jnp.ones(Nz) / (Nz*1.0)),
            transitions=dict(transition_matrix=0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz),
            emissions=dict()
        )
        param_props = dict(
            initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions = dict()
            )
        return (params, param_props)

    def initialize_params(self, key: PRNGKey) -> Tuple[ParamsHMM, ParamPropsHMM]:
        Nz, Ny = self.num_states, self.emission_dim
        params = ParamsHMM(
            initial_dist = jnp.ones(Nz) / (Nz*1.0),
            transition_matrix = 0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz,
            emissions=dict()
        )
        param_props = ParamPropsHMM(
            initial_dist = ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered()),
            transition_matrix = ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            emissions = dict()
        )
        return (params, param_props)

    def filter(self, params: ParamsHMM, emissions: EmissionSeqDiscrete, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        #print('filtering params ', params['initial']['probs'])
        print('filtering params ', params.initial_dist)
        loglik = jnp.sum(emissions) # random
        ntime = emissions.shape[0]
        return HMMPosterior(marginal_loglik = loglik, filtered_probs = jnp.zeros((ntime, self.num_states)))

    def sample(self, params: ParamsHMM, key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeqDiscrete]:
        states = jnp.zeros(num_timesteps, dtype=int)
        #states = jnp.zeros(num_timesteps, dtype=float) # wrong type
        #states = jnp.zeros((num_timesteps, self.num_states), dtype=int) # wrong shape
        #emissions = jnp.zeros((num_timesteps, self.emission_dim)) # wrong shape
        emissions = jnp.zeros(num_timesteps, dtype=int)
        return (states, emissions)

@dataclass
class MyHMMV(HMMVecOutput):

    def initialize_params_old(self, key: PRNGKey) -> Tuple[Params, ParamProps]:
        Nz, Ny = self.num_states, self.emission_dim
        emission_means = jr.normal(key, (self.num_states, self.emission_dim))
        emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        params = dict(
            initial=dict(probs=jnp.ones(Nz) / (Nz*1.0)),
            transitions=dict(transition_matrix=0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz),
            emissions=dict(means=emission_means, covs = emission_covs)
        )
        param_props = dict(
            initial=dict(probs=ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered())),
            transitions=dict(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())),
            emissions = dict(means=ParameterProperties(),
                           covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
            )
        return (params, param_props)

    def initialize_params(self, key: PRNGKey) -> Tuple[ParamsHMM, ParamPropsHMM]:
        Nz, Ny = self.num_states, self.emission_dim
        emission_means = jr.normal(key, (self.num_states, self.emission_dim))
        emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        params = ParamsHMM(
            initial_dist = jnp.ones(Nz) / (Nz*1.0),
            transition_matrix = 0.9 * jnp.eye(Nz) + 0.1 * jnp.ones((Nz, Nz)) / Nz,
            emissions=dict(means=emission_means, covs = emission_covs)
        )
        param_props = ParamPropsHMM(
            initial_dist = ParameterProperties(trainable=False, constrainer=tfb.SoftmaxCentered()),
            transition_matrix = ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            emissions = dict(means=ParameterProperties(),
                           covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
            )
        return (params, param_props)

    def filter(self, params: ParamsHMM, emissions: EmissionSeqVec, inputs: Optional[InputSeq]=None) -> HMMPosterior:
        #print('filtering params ', params['initial']['probs'])
        print('filtering params ', params.initial_dist)
        loglik = jnp.sum(emissions) # random
        ntime = emissions.shape[0]
        return HMMPosterior(marginal_loglik = loglik, filtered_probs = jnp.zeros((ntime, self.num_states)))

    def sample(self, params: ParamsHMM, key: PRNGKey, num_timesteps: int,  inputs: Optional[InputSeq]=None) -> Tuple[StateSeqDiscrete, EmissionSeqVec]:
        states = jnp.zeros(num_timesteps, dtype=int)
        emissions = jnp.zeros((num_timesteps, self.emission_dim))
        return (states, emissions)