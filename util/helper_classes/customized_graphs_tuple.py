from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Union, Optional

import jax.numpy as jnp
from jax import tree_util

ArrayTree = Union[jnp.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


@dataclass
class OneEpochIndependentData:
    n_unique_nodes: int  # with integer dtype
    n_unique_edges: int
    max_node: int
    max_edge: int
    n_edges: int

    @classmethod
    def max(cls, a, b):

        return cls(
            n_unique_nodes=max(
                a.n_unique_nodes, b.n_unique_nodes
            ),  # with integer dtype
            n_unique_edges=max(a.n_unique_edges, b.n_unique_edges),
            max_node=max(a.max_node, b.max_node),
            max_edge=max(a.max_edge, b.max_edge),
            n_edges=max(a.n_edges, b.n_edges),
        )


@dataclass
class EpochIndependentData:
    total: OneEpochIndependentData
    graph: OneEpochIndependentData
    data: OneEpochIndependentData


@dataclass
class BatchDependentData:
    """ Helper class contains all information about the graph used during message-passing"""
    node_type: Optional[ArrayTree]
    node_representations: Optional[ArrayTree]
    edge_representations: Optional[ArrayTree]
    edge_type: Optional[ArrayTree]
    head: Optional[
        jnp.ndarray
    ]  # contains the node_type not the node_representation!!! Saves space to do it like this
    tail: Optional[
        jnp.ndarray
    ]  # contains the node_type not the node_representation!!! Saves space to do it like this
    query_representation: Optional[jnp.ndarray]
    bounding_conditions: Optional[ArrayTree]

    def _tree_flatten(self):
        children = (
            self.node_type,
            self.node_representations,
            self.edge_representations,
            self.edge_type,
            self.head,
            self.tail,
            self.query_representation,
            self.bounding_conditions,
        )  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)



tree_util.register_pytree_node(
    BatchDependentData,
    BatchDependentData._tree_flatten,
    BatchDependentData._tree_unflatten,
)
