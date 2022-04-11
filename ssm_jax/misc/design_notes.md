# API design notes

The goal of these notes is to describe (and agree on!) the common API/style/code that would be the foundation of the methods. 

Several principles guide the design:
- Flexibility/reusability: a function should require exactly what it needs to run. 
- Pure functional: no side effect
- Typed without overloading: for readability, not for logic.


## Gaussian approximated methods

### Model definition

The main user defined function will be the way to compute (or approximate) conditional mean and covariances

```python
@cholesky_model
def function(params_at_time_t):
    # Do stuff with the params
    
    def m(x_t):
        """ Computes :math:`\mathbb{E}[x_{t+1} \mid x_t]`
        """
        return ...

    def chol(x_t):
        """ Computes :math:`\mathbb{C}[x_{t+1} \mid x_t]^{1/2}`
        """
        return ...
    return m, chol

# Otherwise, to allow for logic to be shared, we can also do 

@cholesky_model
def function(params_at_time_t):
    # Do stuff with the params
    
    def m_chol(x_t):
        """ Computes :math:`\mathbb{E}[x_{t+1} \mid x_t]` and :math:`\mathbb{C}[x_{t+1} \mid x_t]^{1/2}`
        """
        return MVNChol(...)

    return m_chol

```

where for instance the `cholesky_model` decorator can be defined as 

```python
class CholeskyModel(NamedTuple):
    mean: Callable
    chol: Union[Callable, np.ndarray]
    is_chol_constant: bool
    
def cholesky_model(function):
    @functool.wraps(function)
    def get_model(*args, **kwargs):
        m, chol = function(*args, **kwargs)
        return CholeskyModel(m, chol, not iscallable(chol))
    return get_model
```

similarly for covariance models

```python
class CovarianceModel(NamedTuple):
    mean: Callable
    cov: Union[Callable, np.ndarray]
    is_cov_constant: bool
    
def covariance_model(function):
    @functool.wraps(function)
    def get_model(*args, **kwargs):
        m, cov = function(*args, **kwargs)
        return CovarianceModel(m, cov, not iscallable(cov))
    return get_model
```

Of course, in some cases, the conditional mean and covariance is not tractable in closed-form so that we need to provide the user with a utility function that takes a complicated (functional at least) model, and returns the (appoximated) conditonal one.

```python
def make_conditional_model(f: Callable[[np.ndarray, np.ndarray, ...], np.ndarray], integration_method, sqrt:bool) -> Union[CovarianceModel, CholeskyModel]:
    # For example here we suppose that `q` is a MVNCov
    if sqrt:
        @covariance_model
        def mean_cov(m_t, cov_t, *args, **kwargs):
            def mean(x_t):
                # Something like this I guess, but we can iterate.
                F, bias, _ = linearize(lambda q_: f(x_t, q_, *args, **kwargs), integration_method, MVNCov(m_t, cov_t))
                return F @ m_t + bias
```

Finally, the user can also pass in LGSSMs directly to a public Kalman API beside the non-linear extensions. This will be done in a completely explicit way, where the user will feed in all matrices and say if they want the sqrt or standard form. That is,

```python
def kalman_filter(ys: np.ndarray, F: np.ndarray, Q: Union[MVNCov, MVNChol], H, R:Union[MVNCov, MVNChol], x0: Union[MVNCov, MVNChol], propagate_first:bool = False):
    ...
```

### Algotithms API

We have already described a (simplified version) of the standard Kalman filter. In practice, we need to consider the parallel and sequential implementations as well as the requirement of returning a log-likelihood or not.


```python
def kalman_filter(ys: np.ndarray, F: np.ndarray, Q: Union[MVNCov, MVNChol], H: np.ndarray, R:Union[MVNCov, MVNChol], x0: Union[MVNCov, MVNChol], propagate_first: bool = False, parallel: bool = False, return_log_likelihood: bool=True) -> Tuple[Union[MVNCov, MVNChol], Optional[float]]:
    ...
```

Some additional considerations can be handled in the future, such as 
1. Masked arrays for (partially) missing observations.
2. Parametrized models: e.g., `F = F(t)` that can be optimized in memory.
3. Diagonal noise models that can be solved more efficiently.


On the other hand, non-linear models will follow a similar API with the additional constraint that the linearization states can be added:

```python
def non_linear_kalman_filter(ys: np.ndarray, F: Union[np.ndarray, CovarianceModel, CholeskyModel], Q: Union[MVNCov, MVNChol], H: Union[np.ndarray, CovarianceModel, CholeskyModel], R:Union[MVNCov, MVNChol], x0: Union[MVNCov, MVNChol], linearization_method: Union[Callable, Tuple[Callable, Callable]], linearization_states: Optional[Union[MVNChol, MVNCov]] = None, propagate_first: bool = False, parallel: bool = False, return_log_likelihood: bool=True) -> Tuple[Union[MVNCov, MVNChol], Optional[float]]:
    ...
```

The same goes for the smoothing methods, that will, however, take filtering results and no observation model. We then need an iterated smoother, which will have the same signature as the filter

```python
def iterated_gaussian_smoother(ys: np.ndarray, F: Union[np.ndarray, CovarianceModel, CholeskyModel], Q: Union[MVNCov, MVNChol], H: Union[np.ndarray, CovarianceModel, CholeskyModel], R:Union[MVNCov, MVNChol], x0: Union[MVNCov, MVNChol], linearization_method: Union[Callable, Tuple[Callable, Callable]], linearization_states: Optional[Union[MVNChol, MVNCov]] = None, propagate_first: bool = False, parallel: bool = False, return_log_likelihood: bool=True) -> Tuple[Union[MVNCov, MVNChol], Optional[float]]:
    ...
```

and will be a `filter-smoother` routine wrapped in a fixed point iteration.

We will also need to think about ensemble Kalman filter but I am not familiar enough to comment.


