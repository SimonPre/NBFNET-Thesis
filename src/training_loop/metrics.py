import pathlib
from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np


# jax.lax.map friendly versions of simple comparator functions
def one_input_greater(input_):
    array_1, array_2 = input_
    return jnp.greater(array_1, array_2)


def one_input_greater_eq(input_):
    array_1, array_2 = input_
    return jnp.greater_equal(array_1, array_2)


def one_input_is_close(input_):
    array_1, array_2 = input_
    return jnp.isclose(array_1, array_2)



def compute_individual_indexes(
    entry_in_batch: list, graph: jnp.ndarray, sp: bool
) -> list:
    """
    Function to compute the indices for the creation of a BCOO sparse array
    Checks given a triple, at which indices in the graph other s,p,? or ?,o,p exist.
    :param entry_in_batch: an individual tripel
    :param graph: the entire graph
    :param sp: flag if function is looking for s,p,? or ?,o,p
    :return: the node entities for which s,p,? or ?,o,p is true
    """
    if sp:
        filter_0 = graph[:, 0] == entry_in_batch[0]
    else:
        filter_0 = graph[:, 2] == entry_in_batch[0]

    filter_1 = graph[:, 1] == entry_in_batch[1]
    filter_ = jnp.all(jnp.stack((filter_0, filter_1), axis=1), axis=1)

    if sp:
        return list(graph[:, 2][filter_])
    else:
        return list(graph[:, 0][filter_])


def compute_mask(
    batches: jnp.ndarray,
    graph: jnp.ndarray,
    batch_size: int,
    n_batches: int,
) -> np.ndarray:
    """
    Wraper for compute_individual_indexes on more than multiple batches
    :param batches: batched data
    :param graph: the entire kg
    :param batch_size: the size of the batch
    :param n_batches: the number of batches
    :return:
        A list holding the indices denoting for which nodes is s,p,? or ?,o,p is true, given the batched input
        e.g.
           [ [0,0,1],  [0,0,3], [0,2,1] ],  [0,0,1]: read batch zero, entry zero, node 1

    """
    indices = []

    for i_batch in range(n_batches):
        for i_entry in range(batch_size // 2):
            indices.append(
                [
                    [i_batch, i_entry, int(x)]
                    for x in compute_individual_indexes(
                        [batches[i_batch][i_entry][0], batches[i_batch][i_entry][1]],
                        graph,
                        sp=True,
                    )
                ]
            )

    for i_batch in range(n_batches):
        for i_entry in range(batch_size // 2, batch_size):
            indices.append(
                [
                    [i_batch, i_entry, int(x)]
                    for x in compute_individual_indexes(
                        [batches[i_batch][i_entry][0], batches[i_batch][i_entry][1]],
                        graph,
                        sp=False,
                    )
                ]
            )
    return np.array([item for sublist in indices for item in sublist])


def load_filter(
    name_of_dataset: str,
    split: str,
    graph: jnp.ndarray,
    s_batched: jnp.ndarray,
    p_batched: jnp.ndarray,
    o_batched: jnp.ndarray,
    batch_size: int,
    n_batches: int,
) -> jnp.ndarray:
    """
    Function checking if a filter for metric computation has allready been computed. If not loads it from disc
    :param name_of_dataset:
    :param split: if validation or test
    :param graph: entire KG
    :param s_batched: batched array holding s
    :param p_batched: batched array holding p
    :param o_batched: batched array holding o
    :param batch_size: size of a batch
    :param n_batches: number of batches
    :return:
        returns the filter as array.
    """
    path = (
        pathlib.Path(__file__).parent.parent.parent
        / f"data/datasets/{name_of_dataset}/{split}_{batch_size}filter.bytes"
    )
    if path.exists():
        with open(str(path), "rb") as data_file:
            byte_data = data_file.read()
        loaded = np.frombuffer(byte_data, dtype=np.int64)
        n_elements = int(len(loaded) / 3)
        filter_indices = np.reshape(loaded, (n_elements, 3))
    else:
        first_s, second_s = jnp.split(s_batched, 2, axis=-1)
        first_p, second_p = jnp.split(p_batched, 2, axis=-1)
        first_o, second_o = jnp.split(o_batched, 2, axis=-1)

        first_half = jnp.stack((first_s, first_p), axis=-1)
        second_half = jnp.stack((second_o, second_p), axis=-1)

        filter_indices = compute_mask(
            jnp.concatenate((first_half, second_half), axis=1),
            graph,
            batch_size,
            n_batches,
        )

        with open(str(path), "wb") as outfile:
            packed = filter_indices.tobytes()
            outfile.write(packed)
    return jnp.array(filter_indices)


def compute_sameness_mask(
    note_to_which_same: jnp.ndarray, batch_size: int, n_batches: int
) -> jnp.ndarray:
    """
    Computes a mask to filter out the true o, from the array of all possible o
    is redundant if filter is used.
    :param note_to_which_same: true o
    :param batch_size: size of a batch
    :param n_batches: number of batches
    :return:
        Returns a mask for filtering out the true o
    """
    return jnp.stack(
        (
            jnp.repeat(jnp.arange(n_batches), batch_size),
            jnp.repeat(
                jnp.arange(batch_size)[jnp.newaxis, :], n_batches, axis=0
            ).flatten(),
            note_to_which_same.flatten(),
        ),
        axis=1,
    )


def compute_metrics_from_better(
    ranks: jnp.ndarray,
    hits_at_n: Sequence[int],
):
    """
    Actual metric computation
    :param ranks: The ranks as computed in the ranking function
    :param hits_at_n: iterable holding valid values for n
    :return:
        an array of length 2 + len(hits_at_n). Contains all metrics
    """
    mr = ranks.mean()
    mrr = (1 / ranks).mean()
    hits = []
    if hits_at_n:
        for i in hits_at_n:
            hits.append(jnp.less_equal(ranks, jnp.array([i])).mean())
    return jnp.array([mr, mrr, *hits])


def generate_compute_metrics(
    graph: jnp.ndarray,
    s_batched: jnp.ndarray,
    p_batched: jnp.ndarray,
    o_batched: jnp.ndarray,
    n_padded: int,
    n_batches: int,
    filter_: bool,
    batch_size: int,
    hits_at_n: Sequence[int],
    name_of_dataset: str,
    split: Literal["validation", "test"],
    max_node: int,
) -> Callable:
    """
    Generator function to compute the validation and test metrics.
    :param graph: the entire kg
    :param s_batched: batched array holding all s
    :param p_batched: batched array holding all p
    :param o_batched: batched array holding all o
    :param n_padded: number of fake edges padded to the graph
    :param n_batches: number of batches
    :param filter_: flag if the results are filterd or not
    :param batch_size: size of the batch
    :param hits_at_n: iterable holding valid values for n
    :param name_of_dataset: name of the dataset
    :param split: flag if validation or test
    :param max_node: largest used node index, needed to determine array shapes.
    :return:
    """
    first_half_o, second_half_o = jnp.split(o_batched, 2, axis=1)
    first_half_s, second_half_s = jnp.split(s_batched, 2, axis=1)

    to_mask_because_same_as_original = compute_sameness_mask(
        note_to_which_same=jnp.concatenate((first_half_o, second_half_s), axis=-1),
        n_batches=n_batches,
        batch_size=batch_size,
    )

    if filter_:
        filter_indices = load_filter(
            name_of_dataset=name_of_dataset,
            split=split,
            graph=graph,
            s_batched=s_batched,
            p_batched=p_batched,
            o_batched=o_batched,
            batch_size=batch_size,
            n_batches=n_batches,
        )

    def compute_metrics(predictions: jnp.ndarray):
        positives = predictions[
            to_mask_because_same_as_original[:, 0],
            to_mask_because_same_as_original[:, 1],
            to_mask_because_same_as_original[:, 2],
        ].reshape(n_batches, batch_size)

        negatives = predictions.at[
            to_mask_because_same_as_original[:, 0],
            to_mask_because_same_as_original[:, 1],
            to_mask_because_same_as_original[:, 2],
        ].set(-jnp.inf)

        if filter_:
            negatives = negatives.at[
                filter_indices[:, 0], filter_indices[:, 1], filter_indices[:, 2]
            ].set(-jnp.inf)
        negatives = jnp.reshape(negatives, ((n_batches) * batch_size, max_node))
        positives = jnp.reshape(positives, ((n_batches) * batch_size, 1))
        if n_padded:
            negatives = negatives[: -2 * n_padded]
            positives = positives[: -2 * n_padded]

        rankings = (positives <= negatives).sum(axis=1) + 1

        return compute_metrics_from_better(
            ranks=rankings,
            hits_at_n=hits_at_n,
        )

    return compute_metrics
