import dataclasses


@dataclasses.dataclass
class ToPad:
    train: int
    valid: int
    test: int
