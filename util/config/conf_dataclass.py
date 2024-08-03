from typing import Sequence, Literal

from pydantic.dataclasses import dataclass

"""
    Multiple pydantic classes and sub classes, used during the reading of an input file.
    Are ment to ensure that no invalid parameters can be supplied
    Do not rule out invalid parameter combinations
"""


@dataclass(frozen=True)
class NegativeSampling:
    n_negative_samples: int
    filter: bool


@dataclass(frozen=True)
class Data:
    dataset: str
    testing_filter: int | None
    remove_easy_edges: bool
    negative_sampling: NegativeSampling


@dataclass(frozen=True)
class Scoring:
    dimensionalities: Sequence[int]
    activation_function: str
    type: Literal["o", "so", "distmult", "transe", "rotate"]
    augment_with_query_embedding: bool


@dataclass(frozen=True)
class MessagePassing:
    indicator_function_as_bounding: bool
    query_dependent_edge_representations: bool
    initial_s_dependent_on_p: bool
    aggregation_function: Literal["sum", "max", "mean", "pna"]
    messanger_function: Literal["transe", "distmult", "rotate"]
    query_embedding_dimensionality: int
    edge_representation_dimensionalities: Sequence[int]
    activation_function: str
    layer_normalization: bool
    message_augmentation: bool
    skip_connection: bool
    new_representation_based_only_on_update: bool
    learned_zero_vector: bool


@dataclass(frozen=True)
class Optimizer:
    optimizer: str
    learning_rate: float
    other_arguments_kwargs: dict | None


@dataclass(frozen=True)
class Training:
    batch_size: int
    mini_batch_size: int
    n_epochs: int
    scoring: Scoring
    message_passing: MessagePassing
    optimizer: Optimizer
    adversarial_temperature: float
    add_self_edges: bool


@dataclass(frozen=True)
class Evaluation:
    hits_at_N: Sequence[int]
    filter: bool
    ranking_scenario: Literal["mean", "pessimistic", "optimistic"]


@dataclass(frozen=True)
class Run:
    save_results: bool
    data: Data
    seed: int
    training: Training
    evaluation: Evaluation


@dataclass(frozen=True)
class Search(Run):
    n_searches: int
    n_sobol_trials: int


@dataclass(frozen=True)
class SearchConfig:
    Search: Search


@dataclass(frozen=True)
class RunConfig:
    run: Run
