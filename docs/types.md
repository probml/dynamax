# Terminology for types

Dynamax uses [jaxtyping](https://github.com/google/jaxtyping), that provides type declarations
for JAX arrays. These declarations can be checked at run time by using other libraries,
such as [beartype](https://github.com/beartype/beartype)
or [typeguard](https://github.com/agronholm/typeguard). However, currently the run-time checking is
disabled, so the declarations are just for documentation purposes.

Following [diffrax](https://docs.kidger.site/diffrax/api/type_terminology/),
our API documentation uses a few convenient shorthands for some types.

- `Scalar` refers to either an `int`, `float`, or a JAX array with shape `()`.
- `PyTree` refers to any PyTree.
- `Array` refers to a JAX array.

---

In addition shapes and dtypes of `Array`s are annotated:

- `Array["dim1", "dim2"]` refers to a JAX array with shape `(dim1, dim2)`, and so on for other shapes.
    - If a dimension is named in this way, then it should match up and be of equal size to the equally-named dimensions of all other arrays passed at the same time.
    - `Array[()]` refers to an array with shape `()`.
    - `...` refers to an arbitrary number of dimensions, e.g. `Array["times", ...]`.
- `Array[bool]` refers to a JAX array with Boolean dtype. (And so on for other dtypes.)
- These are combined via e.g. `Array["dim1", "dim2", bool]`.
- Some arguments may have different shapes, for example, a transition matrix may be constant or time-varying.
For this, we use `Union` types.


