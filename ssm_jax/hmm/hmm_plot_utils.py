# !pip install graphviz
from graphviz import Digraph



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
        table = [f'<TR><TD>{observations[j]}</TD><TD>{"%.2f" % prob}</TD></TR>' for j, prob in
                 enumerate(obs_mat[i])]
        label = f'''<<TABLE><TR><TD BGCOLOR="lightblue" COLSPAN="2">{name}</TD></TR>{''.join(table)}</TABLE>>'''
        dot.node(f's{i}', label=label)

    # Writes transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            dot.edge(f's{i}', f's{j}', label=str('%.2f' % trans_mat[i, j]))
    dot.attr(rankdir='LR')
    # dot.render(file_name, view=True)
    return dot