from collections.abc import Callable

import jax
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial


def segmented_pna(
    values: jnp.array,
    index: jnp.array,
    degree_out: jnp.array,
    n_nodes: int,
    representation_dim: int,
):
    """
    Aggregates a set of messages using the PNA algorythm es explain in the thesis
    Function works on one graph, so one training example including its negatives
    :param values: messages to be aggregated
    :param index: holds o/tail of each edge
    :param degree_out: vector containing the out degree of node
    :param n_nodes: number of nodes in the graph
    :param representation_dim: number of dimensions used in one message, e.g. 32
    :return:
        An array holding all aggregated messages
    """
    # epsilon 1e-6 is a fixed constant in the model
    epsilon = 1e-6

    scale = jnp.log(degree_out)[:, jnp.newaxis, jnp.newaxis]
    scale = scale / jnp.mean(scale)
    mean = jraph.segment_mean(values, index, num_segments=n_nodes)
    sq_mean = jraph.segment_mean(values**2, index, num_segments=n_nodes)
    max_ = jraph.segment_max(values, index, num_segments=n_nodes)
    min_ = jraph.segment_min(values, index, num_segments=n_nodes)
    x = sq_mean - mean**2

    x_clipped = jax.lax.clamp(epsilon, x, jnp.inf)

    std = jnp.sqrt(x_clipped)

    features = jnp.concatenate(
        [
            mean[:, :, jnp.newaxis],
            max_[:, :, jnp.newaxis],
            min_[:, :, jnp.newaxis],
            std[:, :, jnp.newaxis],
        ],
        axis=-1,
    )
    features = jnp.reshape(features, (n_nodes, 4 * representation_dim))

    scales = jnp.concatenate(
        (
            jnp.ones(n_nodes)[:, jnp.newaxis, jnp.newaxis],
            scale,
            1 / scale.clip(min=1e-2),
        ),
        axis=-1,
    )

    update = jnp.reshape(
        features[:, :, None] * scales, (n_nodes, 4 * representation_dim * 3)
    )
    return update


def generate_aggregation_function(
    func_name: str,
    n_nodes: int,
    query_embedding_dim: int,
) -> Callable:
    """
    Generator to create an aggregation function used in message passing.
    :param func_name: name of the used function, can be sum, max, mean or pna
    :param n_nodes: number of nodes in the graph_attributes
    :param query_embedding_dim: number of dimensions used in one message, e.g. 32
    :return: A vectorized, partial function, to aggregate messages to new node representations.
    """

    def agg_function(messages, indexes, degree_out):
        match func_name:
            case "sum":
                aggregation = jax.vmap( # cool feature of jax for easy vectorization
                    Partial(jax.ops.segment_sum, num_segments=n_nodes)
                )
            case "max":
                aggregation = jax.vmap(
                    Partial(jax.ops.segment_max, num_segments=n_nodes)
                )
            case "mean":
                aggregation = jax.vmap(
                    Partial(jraph.segment_mean, num_segments=n_nodes)
                )
            case "pna":
                aggregation = jax.vmap(
                    Partial(
                        segmented_pna,
                        n_nodes=n_nodes,
                        representation_dim=query_embedding_dim,
                    ),
                    in_axes=[0, 0, None],
                )
            case _:
                raise ValueError(
                    "Please provide a valid aggregation function. Currently available are sum,max,mean, and pna."
                )
        if func_name == "pna":
            return aggregation(messages, indexes, degree_out)
        else:
            return aggregation(messages, indexes)

    return agg_function
