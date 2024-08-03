from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.data_prep.batching import generate_batch_creation_function
from src.data_prep.edge_removal import generate_edge_removal
from src.training_loop.negative_sampling import generate_negative_sampling_function
from src.training_loop.negative_sampling import subject_object_inversion
from src.training_loop.train_step import generate_train_step_function
from util.config.conf_dataclass import RunConfig
from util.helper_classes.customized_graphs_tuple import BatchDependentData


@jax.jit
def tree_transpose(list_of_trees, weights):
    """
        Convert a list of trees of identical structure into a single tree of lists. aggregated via the weighted sum
        :param list_of_trees: A list of pytrees
        :param weights: a list of weights for each entry in the list_of_trees
    """
    # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
    def helper(*args):
        sum_ = jnp.sum(
            jnp.concatenate(
                [arg[..., jnp.newaxis] * weights[i] for i, arg in enumerate(args)],
                axis=-1,
            ),
            axis=-1,
        )
        return sum_ / jnp.sum(weights)

    return jax.tree_map(helper, *list_of_trees)


def generate_train_one_epoch(
    config: RunConfig,
    training_data: jnp.ndarray,
    fixed_graph_information,
    model_generator: Callable,
    n_to_pad_for_batch: int,
):
    """
    Wrapper function around the training of one epoch. Helps to keep code more readable
    :param config: the confing object containing all parameters
    :param training_data: the training data as array of triples
    :param fixed_graph_information: helper object holding all unchanging graph characteristics
    :param model_generator: Callable with the ability to generate a model instance
    :param n_to_pad_for_batch: number of tripels which need to be padded for batching
    :return:
        a function which executes one training epoch.
    """
    n_batches = int(
        jnp.ceil(fixed_graph_information.n_edges / config.run.training.batch_size)
    )

    batch_function = generate_batch_creation_function(
        data=training_data,
        n_batches=n_batches,
        batch_size=config.run.training.batch_size,
        max_node=fixed_graph_information.max_node,
        re_shuffel=True,
    )

    edge_removal_function = generate_edge_removal(
        graph_data=training_data,
        # if false, direct edges still will be removed
        easy_edge_removal=config.run.data.remove_easy_edges,
    )

    if config.run.data.negative_sampling.filter:
        mode = "random_filtered"
    else:
        mode = "random"

    sample_space = [
        fixed_graph_information.max_node,
        fixed_graph_information.max_edge,
        fixed_graph_information.max_node,
    ]

    negative_sampling_func = generate_negative_sampling_function(
        sample_space=sample_space,
        n_samples=config.run.data.negative_sampling.n_negative_samples,
        batch_size=config.run.training.batch_size,
        existing_triples=training_data,
        dataset_name=config.run.data.dataset,
        mode=mode,
    )

    one_train_step_function = generate_train_step_function(
        config=config,
        model_generator=model_generator,
        n_negative_samples=config.run.data.negative_sampling.n_negative_samples,
    )

    def train_one_epoch(state, rng_key):
        """Train for 1 epoch on the training set. """
        batch_metrics = []
        keys = jax.random.split(rng_key, n_batches + 2)
        batched_training_data = batch_function(rng_key=keys[-2])
        for i in range(n_batches):

            batch = batched_training_data[i]

            s, p, o = negative_sampling_func(input_triples=batch, random_key=keys[i])

            p = jnp.repeat(
                p[:, None],
                config.run.data.negative_sampling.n_negative_samples + 1,
                axis=1,
            )
            graph_for_batch = edge_removal_function(s=s, p=p, o=o)
            graph_for_batch = jnp.concatenate(
                (
                    graph_for_batch,
                    inverse_triple_array(graph_for_batch, sample_space[1]),
                )
            )
            if config.run.training.add_self_edges:
                self_edges = jnp.stack(
                    (
                        jnp.arange(fixed_graph_information.max_node + 1),
                        jnp.repeat(
                            fixed_graph_information.max_edge,
                            fixed_graph_information.max_node + 1,
                        ),
                        jnp.arange(fixed_graph_information.max_node + 1),
                    )
                ).T
                graph_for_batch = jnp.concatenate((graph_for_batch, self_edges))

            degree_out = (
                jax.ops.segment_sum(
                    jnp.squeeze(jnp.ones(graph_for_batch.shape[0])),
                    jnp.squeeze(graph_for_batch[:, 2]),
                    num_segments=fixed_graph_information.max_node,
                )
                + 1
            )

            s, p, o = subject_object_inversion(
                s=s,
                p=p[:, 1],
                o=o,
                sample_space=sample_space,
            )
            s = s[:, 0]
            batch_size = config.run.training.batch_size

            # to padd graph to a fixed size avoiding jit recompilation
            if config.run.training.add_self_edges:
                n_edges = (
                    fixed_graph_information.n_edges * 2
                    + fixed_graph_information.max_node
                    + 1
                )
            else:
                n_edges = fixed_graph_information.n_edges * 2
            graph_for_batch = jnp.concatenate(
                (
                    graph_for_batch,
                    jnp.repeat(
                        jnp.array(
                            [sample_space[0], sample_space[1] * 2, sample_space[0]]
                        )[jnp.newaxis, :],
                        n_edges - graph_for_batch.shape[0],
                        axis=0,
                    ),
                )
            )

            dependent_graph_attributes = BatchDependentData(
                node_type=None,
                node_representations=None,
                edge_representations=None,
                edge_type=graph_for_batch[:, 1],
                head=graph_for_batch[:, 0],
                tail=graph_for_batch[:, 2],
                query_representation=None,
                bounding_conditions=None,
            )
            if i == (n_batches - 1):
                s = s[: (config.run.training.batch_size - n_to_pad_for_batch)]
                p = p[: (config.run.training.batch_size - n_to_pad_for_batch)]
                o = o[: (config.run.training.batch_size - n_to_pad_for_batch)]
                batch_size = config.run.training.batch_size - n_to_pad_for_batch

            mini_batch_counter = 0
            mini_batch_size = config.run.training.mini_batch_size
            n_mini_batches = max(int(np.ceil(batch_size / mini_batch_size)), 1)
            loss_list = []
            grad_list = []
            weights_for_average = []
            for j in range(n_mini_batches):
                if j == (n_mini_batches - 1):
                    mini_s = s[mini_batch_counter:]
                    mini_p = p[mini_batch_counter:]
                    mini_o = o[mini_batch_counter:]
                    if not (weight := (batch_size % mini_batch_size)):
                        weight = mini_batch_size
                    weights_for_average.append(weight)
                else:
                    mini_s = s[
                        mini_batch_counter : (mini_batch_counter + mini_batch_size)
                    ]
                    mini_p = p[
                        mini_batch_counter : (mini_batch_counter + mini_batch_size)
                    ]
                    mini_o = o[
                        mini_batch_counter : (mini_batch_counter + mini_batch_size)
                    ]
                    weights_for_average.append(mini_batch_size)
                    weight = mini_batch_size

                # hier noch mini badges einfügen

                mini_batch_counter += mini_batch_size

                mini_loss, mini_grads = one_train_step_function(
                    state,
                    dependent_graph_attributes,
                    mini_s,
                    mini_p,
                    mini_o,
                    weight,
                    degree_out,
                )
                loss_list.append(jnp.array([mini_loss]))
                grad_list.append(mini_grads)

            weights_for_average = jnp.array(weights_for_average)
            loss = jnp.sum(weights_for_average * jnp.concatenate(loss_list)) / jnp.sum(
                weights_for_average
            )
            if len(grad_list) == 1:
                state = state.apply_gradients(grads=grad_list[-1])
            else:
                grads = tree_transpose(grad_list, weights_for_average)
                state = state.apply_gradients(grads=grads)

        batch_metrics.append(loss)

        return state, np.mean(batch_metrics), keys[-1]

    return train_one_epoch


def inverse_triple_array(array: jnp.ndarray, max_p) -> jnp.ndarray:
    """
    Helper function to inverse a given spo array with [n_edges X 3]
    :param array: Jax numpy array of shape  [n_edges X (s,p,o)]
    :return: Returns the same array but inversed. New edge types are generated as inverses of originals
    """
    result = jnp.array(
        [
            array[:, 2],
            array[:, 1] + max_p,
            array[:, 0],
        ]
    )
    return result.T
