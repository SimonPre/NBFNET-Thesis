from src.data_prep.compute_padding_sizes import compute_paddings_sizes_for_batching
from src.training_loop.create_model_partials import generate_model_partials
from src.training_loop.train_epoch import generate_train_one_epoch
from src.training_loop.train_state import create_train_state
from src.training_loop.validation_and_test import generate_validation
from util.helper_classes.step_holder import Steps


def generate_train_valid_test(
    config, data, epoch_independent_train_info, initialization_key, max_node_in_test
):
    """
    Another helper class to abstract the creation of the model further
    :param config: config for this run
    :param data: dictionary holding the test, train and validation data
    :param epoch_independent_train_info: Object of helper class containing unchanging graph characteristics
    :param initialization_key: key for random number generator in jax
    :param max_node_in_test: Maximum node in test, assumes all nodes labelled consecutively with an increasing number
    :return:
    """
    padding_sizes = compute_paddings_sizes_for_batching(
        batch_size=config.run.training.batch_size,
        n_triples_train=epoch_independent_train_info.n_edges,
        n_triples_valid=data["valid"].shape[0],
        n_triples_test=data["test"].shape[0],
    )

    model_partials = generate_model_partials(
        config=config,
        epoch_independent_train_info=epoch_independent_train_info,
        max_node_in_test=max_node_in_test,
    )

    train_one_epoch = generate_train_one_epoch(
        config=config,
        training_data=data["train"],
        fixed_graph_information=epoch_independent_train_info,
        model_generator=model_partials.train,
        n_to_pad_for_batch=padding_sizes.train,
    )

    validate = generate_validation(
        config=config,
        data=data,
        validation_model=model_partials.valid,
        max_node=max_node_in_test,
        max_edge=epoch_independent_train_info.max_edge,
        n_to_pad=padding_sizes.valid,
        split="valid",
    )

    test = generate_validation(
        config=config,
        data=data,
        validation_model=model_partials.test,
        max_node=max_node_in_test,
        max_edge=epoch_independent_train_info.max_edge,
        n_to_pad=padding_sizes.train,
        split="test",
    )

    train_state = create_train_state(
        initialization_key,
        config=config,
        model_generator=model_partials.train,
        fixed_train_info=epoch_independent_train_info,
        learning_rate=config.run.training.optimizer.learning_rate,
        optimizer=config.run.training.optimizer.optimizer,
        other_optimizer_arguments=config.run.training.optimizer.other_arguments_kwargs,
    )

    return (
        Steps(
            train_one_epoch=train_one_epoch,
            validate=validate,
            test=test,
        ),
        train_state,
    )

