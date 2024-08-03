import pathlib
from collections import defaultdict

from jax import numpy as jnp
from numpy import genfromtxt

from util.config.config import RunConfig
from util.helper_classes.customized_graphs_tuple import OneEpochIndependentData


def load_and_prepare_data(
    config: RunConfig,
) -> tuple[dict[str, jnp.ndarray], OneEpochIndependentData]:
    """
    Takes a config object, returns data, and helpful information about the data
    :param config: Config object holding all parameters of a given trial
    :return:
        Returns a dictionary holding test, train, and validation data. As well as a helper object with information
        about test, train and validation.
    """

    data_triples = load_data_set_to_df(config.run.data.dataset)

    # if testing_filter ist used all nodes with an id > x can be filtered out from train
    if config.run.data.testing_filter:

        data_triples["train"] = data_triples["train"][
            jnp.logical_and(
                data_triples["train"][:, 0] < config.run.data.testing_filter,
                data_triples["train"][:, 2] < config.run.data.testing_filter,
            )
        ]
    epoch_independent_training_info = compute_partial_epoch_independent_info(
        data_triples["train"], data_triples["test"], data_triples["valid"]
    )
    return data_triples, epoch_independent_training_info


def compute_partial_epoch_independent_info(
    data_train: jnp.ndarray, data_test: jnp.ndarray, data_valid: jnp.ndarray
) -> OneEpochIndependentData:
    """
    Computes a helper object which holds information about each dataset
    :param data_train: array of train triples
    :param data_test: array of test triples
    :param data_valid: array of validation triples
    """
    max_edge = max(
        jnp.max(data_train[:, 1]), jnp.max(data_test[:, 1]), jnp.max(data_valid[:, 1])
    )
    max_node = max(
        jnp.max(jnp.concatenate((data_test[:, 0], data_test[:, 2]))),
        jnp.max(jnp.concatenate((data_train[:, 0], data_train[:, 2]))),
        jnp.max(jnp.concatenate((data_valid[:, 0], data_valid[:, 2]))),
    )
    return OneEpochIndependentData(
        n_unique_nodes=int(max_node + 1) + 1,
        n_unique_edges=int(max_edge + 1) + 1,
        max_node=int(max_node) + 1,
        max_edge=int(max_edge) + 1,
        n_edges=int(data_train.shape[0]),
    )


def load_data_set_to_df(name_of_data: str) -> dict[str, jnp.ndarray]:
    """
    Function to load dataset from train folder.
    :param name_of_data: name of dataset Needs to conform to the same naming convention as in libkge
    :return:
        Returns dict containing training and test data
    """
    path = pathlib.Path(__file__).parent / f"datasets/{name_of_data}"
    return_dict = defaultdict(jnp.ndarray)
    for element in ["test", "valid", "train"]:
        path_to_element = path / f"{element}.del"
        return_dict[element] = jnp.array(
            genfromtxt(path_to_element, delimiter="\t", dtype=int)
        )
    return return_dict
