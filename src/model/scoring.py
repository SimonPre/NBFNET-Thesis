from collections.abc import Callable
from typing import Sequence

import flax.linen as nn


class MLP(nn.Module):
    """
        Class defining a Multi Layer Perceptron
    """
    activation_function: Callable
    n_neurons_per_layer: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, n_neuron in enumerate(self.n_neurons_per_layer):
            x = nn.Dense(n_neuron)(x)
            if i != len(self.n_neurons_per_layer) - 1:
                x = self.activation_function(x)
        return x


def generate_scoring_activation_function(function_name: str):
    """
    Helper function returning a scoring function
    :param function_name: name of the activation function
    :return: A scoring function if the provided name is the name of scoring function part of the linen package.
    """
    try:
        return eval(f"nn.{function_name}")
    except AttributeError:
        raise ValueError(f"{function_name} is not a valid linen activation function")
