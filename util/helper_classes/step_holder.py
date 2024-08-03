import dataclasses
from collections.abc import Callable


@dataclasses.dataclass
class Steps:
    test: Callable
    validate: Callable
    train_one_epoch: Callable
