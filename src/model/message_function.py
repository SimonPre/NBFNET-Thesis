from collections.abc import Callable

import jax.numpy as jnp
from jax.tree_util import Partial


def rotate(node_reps: jnp.ndarray, edge_reps: jnp.ndarray) -> jnp.ndarray:
    """
    Implementation of the rotate function for message passing
    :param edge_reps: Edge representations based on which messages are computed
    :param node_reps: Representation of source nodes
    :return: An array of dimensionality n_edges x embedding_dim
    """
    node_re, node_im = jnp.split(node_reps, 2, axis=-1)
    edge_re, edge_im = jnp.split(edge_reps, 2, axis=-1)
    message_re = node_re * edge_re - node_im * edge_im
    message_im = node_re * edge_im + node_im * edge_re
    message = jnp.concatenate((message_re, message_im), axis=-1)
    return message


def update_edges(
    edge_representations: jnp.array,
    node_representation: jnp.array,
    source: jnp.array,
    update_func: Callable,
    batch_size: int,
):
    """
    Function which combines edge representation and source representations to obtain messages
    :param edge_representations: Edge representations based on which messages are computed
    :param node_representation: Representation of source nodes
    :param source: List of sources for each edge, contains indices e.g  (batchsize X [1,0,3,2]
    :param update_func: function to combine the node and the edge representation
    :param batch_size: number of graphs contained in a given batch
    :return: An array holding the message on each edge, dimensionality: (batchsize * n_edges * edge_representation_dim)
    """
    # works but no idea why! See:
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    relevant_nodes = node_representation[jnp.arange(batch_size)[:, None], source]
    return update_func(edge_representations, relevant_nodes)


def generate_messanger_function(func_name: str, batchsize: int) -> Callable:
    """
    Generates a messanger function which aggregates the source nodes representation with the edge representation.
    :param func_name: name of the function used to generate the message
    :param batchsize:
    :return: A partial function for message computation given a fixed batch size.
    """
    match func_name:
        case "transe":
            func = jnp.add
        case "distmult":
            func = jnp.multiply
        case "rotate":
            func = rotate
        case _:
            raise ValueError(
                "Please provide a valid messanger function. Currently available are transe, distmult, and rotate."
            )
    return Partial(update_edges, update_func=func, batch_size=batchsize)
