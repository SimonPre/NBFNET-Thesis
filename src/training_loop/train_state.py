from collections.abc import Callable

import jax
from flax.training import train_state

from util.config.conf_dataclass import RunConfig
from util.helper_classes.customized_graphs_tuple import BatchDependentData


def create_fake_model_input_for_init(config: RunConfig, key, fixed_graph_attributes):
    """
    Helper function to create some dummy input, which is needed for model initialization. Does not really influence
    the results, but is needed because of automatic shape inference.
    :param config: the config object holding all parameters
    :param key: key used for random number generation
    :param fixed_graph_attributes: helper object with information about unchanging graph attributes
    :return:
        returns a valid model input, with meaningless content but correct shapes
    """
    batch_dependent_graph_attributes = BatchDependentData(
        node_type=None,
        node_representations=None,
        edge_representations=None,
        edge_type=jax.random.randint(
            key,
            shape=(fixed_graph_attributes.n_edges * 2,),
            maxval=fixed_graph_attributes.max_edge * 2,
            minval=0,
        ),
        head=jax.random.randint(
            key,
            shape=(fixed_graph_attributes.n_edges * 2,),
            maxval=fixed_graph_attributes.max_node,
            minval=0,
        ),
        tail=jax.random.randint(
            key,
            shape=(fixed_graph_attributes.n_edges * 2,),
            maxval=fixed_graph_attributes.max_node,
            minval=0,
        ),
        query_representation=None,
        bounding_conditions=None,
    )
    key1, key2, key3 = jax.random.split(key, 3)
    s = jax.random.randint(
        key1,
        shape=(config.run.training.mini_batch_size,),
        minval=0,
        maxval=fixed_graph_attributes.max_node + 1,
    )

    p = jax.random.randint(
        key2,
        shape=(config.run.training.mini_batch_size,),
        minval=0,
        maxval=(fixed_graph_attributes.max_edge * 2) + 1,
    )
    o = jax.random.randint(
        key3,
        shape=(
            config.run.training.mini_batch_size,
            config.run.data.negative_sampling.n_negative_samples + 1,
        ),
        minval=0,
        maxval=fixed_graph_attributes.max_node,
    )
    degree_out = jax.random.randint(
        key3, shape=(fixed_graph_attributes.max_node,), minval=0, maxval=60
    )

    return (
        batch_dependent_graph_attributes,
        s,
        p,
        o,
        config.run.training.mini_batch_size,
        degree_out,
    )


def create_train_state(
    key,
    config: RunConfig,
    model_generator: Callable,
    fixed_train_info,
    learning_rate: float,
    optimizer: str,
    other_optimizer_arguments: dict | None,
):
    """
    Creates train state object, managing the transfer of information across epochs.
    :param key: key used for random number generation
    :param config: the config object holding all parameters
    :param model_generator: Callable with the ability to generate a model instance
    :param fixed_train_info:   object holding all unchanging graph characteristics
    :param learning_rate: learning rate for the optimizer
    :param optimizer: name of the optimizer as string
    :param other_optimizer_arguments: other arguments passed as kwargs to the optimizer
    :return:
        a train state object
    """
    model = model_generator()
    params = model.init(
        key, *create_fake_model_input_for_init(config, key, fixed_train_info)
    )
    opt = generate_optimizer(
        optimizer, learning_rate, other_arguments=other_optimizer_arguments
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


def generate_optimizer(name: str, learning_rate: float, other_arguments: dict):
    """
    Helper function to generate an optimizer function from a string
    :param name:  name of the optimizer as string
    :param learning_rate: learning rate for the optimizer
    :param other_arguments: other arguments passed as kwargs to the optimizer
    :return:
    """
    try:
        import optax

        if other_arguments:
            return eval(
                f"optax.{name}(learning_rate={learning_rate},**{other_arguments})"
            )
        else:
            return eval(f"optax.{name}(learning_rate={learning_rate})")
    except AttributeError:
        raise ValueError(f"{name} is not a valid linen activation function")