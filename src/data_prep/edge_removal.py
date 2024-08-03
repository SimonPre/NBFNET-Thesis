import jax.numpy as jnp
from jax.tree_util import Partial


def remove_easy_edges(
    graph_data: jnp.ndarray,
    s: jnp.ndarray,
    p: jnp.ndarray,
    o: jnp.ndarray,
):
    """
    Function removing, the edges in the training batch from the graph. Removes all entries in the graph with same
    s and o.
    :param graph_data: as list of triples
    :param s: array of subjects
    :param p: array of predicates  #to used but added for easier interchangeability with other removal function
    :param o: array of objects
    :return:
        A graph from which all one hope connections between training subjects and objects have been removed.
    """
    so_b = jnp.stack((s.flatten(), o.flatten()), axis=-1)
    so_g = graph_data[:, [0, 2]]
    # https://stackoverflow.com/questions/74154196/multidimensional-jax-isin
    mask1 = (so_g[:, None] == so_b[None, :]).all(-1).any(-1)
    os_g = jnp.stack((so_g[:, 1], so_g[:, 0]), axis=-1)
    mask2 = (os_g[:, None] == so_b[None, :]).all(-1).any(-1)
    mask = jnp.logical_or(mask1, mask2)
    return graph_data[~mask]


def remove_direct_connection_from_graph(
    graph_data: jnp.ndarray,
    s: jnp.ndarray,
    p: jnp.ndarray,
    o: jnp.ndarray,
):
    """
    Function removing, the edges in the training batch from the graph. Only if exact match between edge and graph
    :param graph_data: as list of triples
    :param s: array of subjects
    :param p: array of predicates
    :param o: array of objects
    :return:
        A graph from which all training triples have been removed.
    """
    all_relevant_combinations = jnp.stack((s.flatten(), p.flatten(), o.flatten()))
    mask = (graph_data[:, None] == all_relevant_combinations.T[None, :]).all(-1).any(-1)
    return graph_data[~mask]


def generate_edge_removal(graph_data: jnp.ndarray, easy_edge_removal: bool):
    """
    Generator function producing a nice partial of the edge removal function for later use.
    :param graph_data: as list of triples
    :param easy_edge_removal: Flag indicating if one-hopes are removed or only exact spo matches.
    :return:
        A partial of the appropriate edge removal function
    """
    if easy_edge_removal:
        return Partial(remove_easy_edges, graph_data=graph_data)
    else:
        return Partial(remove_direct_connection_from_graph, graph_data=graph_data)

