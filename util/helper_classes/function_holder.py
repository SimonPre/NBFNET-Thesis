from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionHolder:
    "Helper class to keep the functions together"
    augmentation_function: Optional[Callable]
    message_function: Callable
    aggregation_function: Callable
    batching_function: Callable
    negative_sampling_function: Callable
    edge_removal_function: Callable
    loss_function: Callable
