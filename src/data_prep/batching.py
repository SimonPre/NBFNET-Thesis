from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import Partial


def batch_creation(
    rng_key: jnp.ndarray,
    data: jnp.ndarray,
    n_batches: int,
    batch_size: int,
    max_node: int,
    re_shuffel: bool,
) -> jnp.ndarray:
    """
    Helper function to create batches given jnp.array
    :param rng_key: RNG KEY used as statring point for random number generator
    :param data: dataset consisting of triples
    :param n_batches: number of batches
    :param batch_size: target batch size
    :param max_node: Node with largest index. I.e the virtual node added for padding to the graph
    :param re_shuffel: Flag if dataset should be shuffled before batch creation
    :return:
        Returns a padded array. Padded entries are allways the last entries in the last batch
    """
    if re_shuffel:
        data = jax.random.permutation(rng_key, data, axis=0)
    n_triples_to_add = (batch_size - jnp.shape(data)[0]) % batch_size
    padding_triples = jnp.repeat(
        jnp.array(
            [
                [
                    max_node, # max_node is the padding node
                    -99, # padded here with -99 but does not matter, never makes it into message passing
                    max_node,
                ]
            ]
        ),
        n_triples_to_add,
        axis=0,
    )
    padded_data = jnp.concatenate((data, padding_triples))
    return jnp.reshape(padded_data, (n_batches, batch_size, 3))


def generate_batch_creation_function(
    data: jnp.ndarray,
    n_batches: int,
    batch_size: int,
    max_node: int,
    re_shuffel: bool,
) -> Callable:
    """
    Generator function for batch creation. Usefull for repeated batch creation in every epoch
    :param data: dataset consisting of triples
    :param n_batches: number of batches
    :param batch_size: target batch size
    :param max_node: Node with largest index. I.e the virtual node added for padding to the graph
    :param re_shuffel: Flag if dataset should be shuffled before batch creation
    :return:
        A compiled partial of the batch creation function.
    """
    return jax.jit(
        Partial(
            batch_creation,
            data=data,
            n_batches=n_batches,
            batch_size=batch_size,
            max_node=max_node,
            re_shuffel=re_shuffel,
        )
    )
