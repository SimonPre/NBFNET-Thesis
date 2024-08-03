from src.model.model import create_model_partial
from util.config.config import RunConfig
from util.helper_classes.partial_holder import ModelPartials


def generate_model_partials(
    config: RunConfig, epoch_independent_train_info, max_node_in_test
) -> ModelPartials:
    """
    Helper function, creating all model partials for train, test and validation
    :param config: config for this run
    :param epoch_independent_train_info: Object of helper class containing unchanging graph characteristics
    :param max_node_in_test: Maximum node in test, assumes all nodes labelled consecutively with an increasing number
    :return:
    """
    return ModelPartials(
        train=create_model_partial(
            config,
            epoch_independent_train_info,
            mode="train",
            n_nodes=max_node_in_test + 1, # + 1 because padding
        ),
        valid=create_model_partial(
            config,
            epoch_independent_train_info,
            mode="validation",
            n_nodes=max_node_in_test + 1, # + 1 because padding
        ),
        test=create_model_partial(
            config,
            epoch_independent_train_info,
            mode="validation",
            n_nodes=max_node_in_test + 1, # + 1 because padding
        ),
    )
