"""
Categorial Regression hidden Markov model (HMM) with state-dependent weights and input-driven state transitions.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import jax.random as jr
from jaxtyping import Array, Float, Int, PyTree
import optax

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.categorical_glm_hmm import ParamsCategoricalRegressionHMMEmissions, CategoricalRegressionHMMEmissions

from dynamax.hidden_markov_model.models.inputdriven_linreg_hmm import InputDrivenHMMInitialState, ParamsInputDrivenHMMInitialState
from dynamax.hidden_markov_model.models.inputdriven_linreg_hmm import InputDrivenHMMTransitions, ParamsInputDrivenHMMTransitions


class ParamsInputDrivenCategoricalRegressionHMM(NamedTuple):
    """Parameters for an input-driven categorical regression HMM."""
    initial: ParamsInputDrivenHMMInitialState
    transitions: ParamsInputDrivenHMMTransitions
    emissions: ParamsCategoricalRegressionHMMEmissions


class InputDrivenCategoricalRegressionHMM(HMM):
    r"""An HMM whose emissions come from a categorical regression with state-dependent weights and
    initial-state and transition distributions are driven by an external input, rather than fixed.

    The initial distribution (see `InputDrivenHMMInitialState`) and the transition
    distribution (see `InputDrivenHMMTransitions`) are both multinomial logistic
    regressions on the input,

    $$p(z_1 \mid u_1, \theta) = \mathrm{Cat}(z_1 \mid \mathrm{softmax}(W^{\mathsf{init}} u_1 + b^{\mathsf{init}}))$$

    $$p(z_t \mid z_{t-1}, u_t, \theta) = \mathrm{Cat}(z_t \mid \mathrm{softmax}(W^{\mathsf{trans}}_{z_{t-1}} u_t + b^{\mathsf{trans}}_{z_{t-1}}))$$

    The emission distribution is a state-dependent linear regression, as in `CategoricalRegressionHMM`,

    $$p(y_t \mid z_t, u_t, \theta) = \mathrm{Cat}(y_{t} \mid \mathrm{softmax}(W^{\mathsf{emis}}_{z_t} u_t + b^{\mathsf{emis}}_{z_t}))$$

    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param emission_dim: emission dimension $N$
    :param m_step_optimizer: ``optax`` optimizer, like Adam, used for the transition M-step.
    :param m_step_num_iters: number of optimizer steps per M-step.
    """

    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 50):
        self.num_classes = num_classes
        self.input_dim = input_dim
        initial_component = InputDrivenHMMInitialState(num_states, input_dim)
        transition_component = InputDrivenHMMTransitions(num_states, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        emission_component = CategoricalRegressionHMMEmissions(num_states, num_classes, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return the shape of the input."""
        return (self.input_dim,)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   transition_weights: Optional[Float[Array, "num_states num_states input_dim"]] = None,
                   transition_biases: Optional[Float[Array, "num_states num_states"]] = None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Currently only "prior" is supported.
            transition_weights: manually specified transition weights.
            transition_biases: manually specified transition biases.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, weights=transition_weights, biases=transition_biases)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key=key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases)
        return ParamsInputDrivenCategoricalRegressionHMM(**params), ParamsInputDrivenCategoricalRegressionHMM(**props)
