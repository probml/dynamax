"""
This demo does MAP estimation of an HMM using gradient-descent algorithm applied to the log marginal likelihood.
It includes
1. Mini Batch Gradient Descent
2. Full Batch Gradient Descent
3. Stochastic Gradient Descent
Author: Aleyna Kara(@karalleyna)
""" """
This demo does MAP estimation of an HMM using gradient-descent algorithm applied to the log marginal likelihood.
It includes
1. Mini Batch Gradient Descent
2. Full Batch Gradient Descent
3. Stochastic Gradient Descent
Author: Aleyna Kara(@karalleyna)
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from graphviz import Digraph
from jax import jit
from jax import lax
from jax import nn
from jax import random
from jax import value_and_grad
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.models import BaseHMM


def hmm_plot_graphviz(trans_mat, obs_mat, states=[], observations=[]):
    """
    Visualizes HMM transition matrix and observation matrix using graphhiz.
    Parameters
    ----------
    trans_mat, obs_mat, init_dist: arrays
    states: List(num_hidden)
        Names of hidden states
    observations: List(num_obs)
        Names of observable events
    Returns
    -------
    dot object, that can be displayed in colab
    """

    n_states, n_obs = obs_mat.shape

    dot = Digraph(comment='HMM')
    if not states:
        states = [f'State {i + 1}' for i in range(n_states)]
    if not observations:
        observations = [f'Obs {i + 1}' for i in range(n_obs)]

    # Creates hidden state nodes
    for i, name in enumerate(states):
        table = [f'<TR><TD>{observations[j]}</TD><TD>{"%.2f" % prob}</TD></TR>' for j, prob in enumerate(obs_mat[i])]
        label = f'''<<TABLE><TR><TD BGCOLOR="lightblue" COLSPAN="2">{name}</TD></TR>{''.join(table)}</TABLE>>'''
        dot.node(f's{i}', label=label)

    # Writes transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            dot.edge(f's{i}', f's{j}', label=str('%.2f' % trans_mat[i, j]))
    dot.attr(rankdir='LR')
    # dot.render(file_name, view=True)
    return dot


def hmm_sample_minibatches(sequences, batch_size):
    n_seq = len(sequences)
    for idx in range(0, n_seq, batch_size):
        yield sequences[idx:min(idx + batch_size, n_seq)]


def hmm_fit_gradient_descent(hmm, emissions, optimizer, batch_size=1, num_iters=50, key=random.PRNGKey(0)):
    cls = hmm.__class__
    hypers = hmm.hyperparams

    params = hmm.unconstrained_params
    opt_state = optimizer.init(params)

    num_complete_batches, leftover = jnp.divmod(len(emissions), batch_size)
    num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

    def loss(params, batch_emissions):
        hmm = cls.from_unconstrained_params(params, hypers)
        f = lambda emissions: -hmm.marginal_log_prob(emissions) / len(emissions)
        return vmap(f)(batch_emissions).mean()

    loss_grad_fn = jit(value_and_grad(loss))

    def train_step(carry, key):
        perm = random.permutation(key, len(emissions))
        _emissions = emissions[perm]
        sample_generator = hmm_sample_minibatches(_emissions, batch_size)

        def opt_step(carry, i):
            params, opt_state = carry
            batch = next(sample_generator)
            val, grads = loss_grad_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), val

        state, losses = lax.scan(opt_step, carry, jnp.arange(num_batches))
        return state, losses.mean()

    keys = random.split(key, num_iters)
    (params, _), losses = lax.scan(train_step, (params, opt_state), keys)

    losses = losses.flatten()
    hmm = cls.from_unconstrained_params(params, hypers)

    return hmm, losses


@register_pytree_node_class
class CategoricalHMM(BaseHMM):

    def __init__(self, initial_logits, transition_logits, emission_logits):
        num_states = transition_logits.shape[-1]

        # Check shapes
        assert initial_logits.shape == (num_states,)
        assert transition_logits.shape == (num_states, num_states)

        # Construct the  distribution objects
        self._initial_distribution = tfd.Categorical(logits=initial_logits)
        self._transition_distribution = tfd.Categorical(logits=transition_logits)
        self._emission_distribution = tfd.Categorical(logits=emission_logits)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = random.split(key, 3)
        initial_probs = random.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = random.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = random.dirichlet(key3, jnp.ones(emission_dim), (num_states,))
        return cls(initial_probs, transition_matrix, emission_probs)

    # Properties to get various parameters of the model
    @property
    def emission_distribution(self):
        return self._emission_distribution

    @property
    def initial_probabilities(self):
        return self._initial_distribution.probs_parameter()

    @property
    def emission_probs(self):
        return self._emission_distribution.probs_parameter()

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs_parameter()

    @property
    def initial_logits(self):
        return self._initial_distribution.logits_parameter()

    @property
    def transition_logits(self):
        return self._transition_distribution.logits_parameter()

    @property
    def emission_logits(self):
        return self.emission_distribution.logits_parameter()

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (self.initial_logits, self.transition_logits, self.emission_logits)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)


def init_random_categorical_hmm(key, sizes):
    """
    Initializes the components of CategoricalHMM from normal distibution
    Parameters
    ----------
    key : array
        Random key of shape (2,) and dtype uint32
    sizes: List
      Consists of number of hidden states and observable events, respectively
    Returns
    -------
    * CategoricalHMM
    """
    initial_key, transition_key, emission_key = random.split(key, 3)
    num_hidden_states, num_obs = sizes
    return CategoricalHMM(random.normal(initial_key, (num_hidden_states,)),
                          random.normal(transition_key, (num_hidden_states, num_hidden_states)),
                          random.normal(emission_key, (num_hidden_states, num_obs)))


def main():
    key = random.PRNGKey(0)
    init_key, sample_key, train_key = random.split(key, 3)

    initial_probabilities = jnp.array([1, 1]) / 2

    # state transition matrix
    transition_matrix = jnp.array([[0.95, 0.05], [0.10, 0.90]])

    # observation matrix
    emission_probs = jnp.array([
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],  # fair die
        [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 5 / 10]  # loaded die
    ])

    casino = CategoricalHMM(initial_probabilities, transition_matrix, emission_probs)
    sample_fn = vmap(casino.sample, in_axes=(0, None))
    n, num_timesteps = 4, 5000
    keys = random.split(sample_key, n)
    _, emissions = sample_fn(keys, num_timesteps)

    sizes = emission_probs.shape
    hmm = init_random_categorical_hmm(init_key, sizes)
    num_iters = 400
    learning_rate = 1e-2
    momentum = 0.95
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)

    # Mini Batch Gradient Descent
    batch_size = 2
    hmm_mb, losses_mbgd = hmm_fit_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters, train_key)

    # Full Batch Gradient Descent
    batch_size = n
    hmm_fb, losses_fbgd = hmm_fit_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters, train_key)
    # Stochastic Gradient Descent
    batch_size = 1
    hmm_sgd, losses_sgd = hmm_fit_gradient_descent(hmm, emissions, optimizer, batch_size, num_iters, train_key)

    losses = [losses_sgd, losses_mbgd, losses_fbgd]

    titles = ["Stochastic Gradient Descent", "Mini Batch Gradient Descent", "Full Batch Gradient Descent"]

    dict_figures = {}
    for loss, title in zip(losses, titles):
        filename = title.replace(" ", "_").lower()
        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_title(f"{title}")
        dict_figures[filename] = fig
        plt.savefig(f"{filename}.png", dpi=300)

    dotfile = hmm_plot_graphviz(hmm_sgd.transition_matrix, hmm_sgd.emission_probs)
    dotfile.render("hmm-casino-dot")


if __name__ == "__main__":
    main()
