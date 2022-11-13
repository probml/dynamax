
from jaxtyping import  jaxtyped
#from beartype import beartype as typechecker
from typeguard import typechecked as typechecker
import inspect

def jaxtyped2(fn):
    if inspect.isclass(fn):
        init = jaxtyped2(fn.__init__)
        fn.__init__ = init
        return fn
    else:
        jaxtyped(fn) # existing implementation
       

@jaxtyped2
@typechecker
class LinearGaussianSSM():
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=True
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias


   
    
@typechecker
class LinearGaussianConjugateSSM(LinearGaussianSSM):
    def __init__(self,
                 state_dim,
                 emission_dim,
                 input_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
        has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)




   