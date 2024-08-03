import dataclasses
from collections.abc import Callable


@dataclasses.dataclass
class ModelPartials:
    test: Callable
    valid: Callable
    train: Callable
