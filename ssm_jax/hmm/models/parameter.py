import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Parameter:

    def __init__(self, value, is_frozen=False, bijector=None):
        self.value = value
        self.is_frozen = is_frozen
        self.bijector = bijector if bijector is not None else tfb.Identity()

    def __repr__(self):
        return f"Parameter(value={self.value}, " \
               f"is_frozen={self.is_frozen}, " \
               f"bijector={self.bijector})"

    @property
    def unconstrained_value(self):
        return self.bijector(self.value)

    def tree_flatten(self):
        children = (self.value,)
        aux_data = self.is_frozen, self.bijector
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)
