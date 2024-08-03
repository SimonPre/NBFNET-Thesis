from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.tree_util import Partial

from src.data_prep.batching import generate_batch_creation_function
from src.training_loop.metrics import generate_compute_metrics
from src.training_loop.negative_sampling import generate_negative_sampling_function
from src.training_loop.negative_sampling import subject_object_inversion
from src.training_loop.train_epoch import inverse_triple_array
from util.config.conf_dataclass import RunConfig
from util.helper_classes.customized_graphs_tuple import BatchDependentData


def generate_validation(
    data: jnp.ndarray,
    config: RunConfig,
    max_node,
    max_edge,
    validation_model: Callable,
    n_to_pad: int,
    split: Literal["valid", "test"],
) -> Callable:
    """
    Generator function to creat a validation and test function.
    :param data: array of validation / test triples
    :param config: the config object holding all parameters
    :param max_node: highest node index,
    :param max_edge:  highest edge index
    :param validation_model: instance of the model to be validated
    :param n_to_pad: number of entries which need to be padded to the test / validation data
    :param split: flag if test or validation data is used
    :return:
        a test / validation function performing test or validation
    """
    n_batches = int(
        jnp.ceil(data[split].shape[0] * 2 / config.run.training.mini_batch_size)
    )

    batch_function = generate_batch_creation_function(
        data=data[split],
        n_batches=n_batches,
        batch_size=config.run.training.mini_batch_size // 2,
        max_node=max_node,
        re_shuffel=False,
    )

    batches = batch_function(data[split])
    batches_augmented = jnp.concatenate((batches, batches), axis=1)

    map_second_argument = Partial(jax.vmap, axis_name="input_triples")
    sample_space = [max_node, max_edge, max_node]

    model_graph = jnp.concatenate(
        (data["train"], inverse_triple_array(data["train"], sample_space[1]))
    )

    if config.run.training.add_self_edges:
        self_edges = jnp.stack(
            (
                jnp.arange(max_node + 1),
                jnp.repeat(max_edge, max_node + 1),
                jnp.arange(max_node + 1),
            )
        ).T
        model_graph = jnp.concatenate((model_graph, self_edges))

    batched_s = batches_augmented[:, :, 0]
    batched_p = batches_augmented[:, :, 1]
    batched_o = batches_augmented[:, :, 2]

    compute_metrics = generate_compute_metrics(
        graph=jnp.concatenate((data["train"], data["valid"], data["test"])),
        s_batched=batched_s,
        p_batched=batched_p,
        n_padded=n_to_pad,
        n_batches=n_batches,
        o_batched=batched_o,
        filter_=config.run.evaluation.filter,
        batch_size=config.run.training.mini_batch_size,
        hits_at_n=config.run.evaluation.hits_at_N,
        name_of_dataset=config.run.data.dataset,
        split=split,
        max_node=max_node,
    )

    subject_object_inversion_v = jax.vmap(
        subject_object_inversion, in_axes=[0, 0, 0, None]
    )
    batched_s, batched_p, batched_o_true = subject_object_inversion_v(
        batched_s, batched_p, batched_o, sample_space
    )

    negative_sampling_function = map_second_argument(
        Partial(
            generate_negative_sampling_function(
                sample_space=sample_space,
                n_samples=max_node,
                batch_size=config.run.training.mini_batch_size,
                existing_triples=None,
                mode="all",
                dataset_name=config.run.data.dataset,
            ),
            random_key=jax.random.PRNGKey(0),
        )
    )

    batches_augmented = jnp.concatenate(
        (batched_s[:, :, None], batched_p[:, :, None], batched_o_true[:, :, None]),
        axis=-1,
    )
    a, b = jnp.split(batches_augmented[-1], 2, axis=0)
    n_non_padded = int(config.run.training.mini_batch_size / 2 - n_to_pad)
    batches_augmented = batches_augmented.at[-1].set(
        jnp.concatenate(
            (a[:n_non_padded], b[:n_non_padded], a[n_non_padded:], b[n_non_padded:])
        )
    )
    (batched_s, batched_p, batched_o) = negative_sampling_function(
        input_triples=batches_augmented
    )

    batched_s = batched_s[:, :, 0]

    degree_out = (
        jax.ops.segment_sum(
            jnp.squeeze(jnp.ones(model_graph.shape[0])),
            jnp.squeeze(model_graph[:, 2]),
            num_segments=max_node,
        )
        + 1
    )

    graph_data = BatchDependentData(
        node_type=None,
        node_representations=None,
        edge_representations=None,
        edge_type=model_graph[:, 1],
        head=model_graph[:, 0],
        tail=model_graph[:, 2],
        query_representation=None,
        bounding_conditions=None,
    )

    model = validation_model()

    @jax.tree_util.Partial(jax.jit, static_argnames="batch_size")
    def compute_probs(param, graph_data, s, p, o, batch_size, degree_out):
        return model.apply(param, graph_data, s, p, o, batch_size, degree_out)

    def validate(param: FrozenDict):
        probs = []
        negatives = jnp.concatenate(
            (
                jnp.arange(max_node),
                jnp.repeat(
                    jnp.array([0]),
                    config.run.data.negative_sampling.n_negative_samples
                    - (
                        (max_node)
                        % config.run.data.negative_sampling.n_negative_samples
                    ),
                ),
            )
        )

        for i in range(n_batches):
            s = batched_s[i]
            p = batched_p[i]

            probs_inner = compute_probs(
                param,
                graph_data,
                s,
                p,
                negatives,
                config.run.training.mini_batch_size,
                degree_out,
            )
            probs_inner[-1] = probs_inner[-1][
                :,
                :,
                : (max_node % config.run.data.negative_sampling.n_negative_samples),
            ]

            probs.append(jnp.concatenate(probs_inner, axis=-1))
        probs = jnp.concatenate(probs, axis=0)
        return compute_metrics(probs)

    return validate
