import math
from collections.abc import Callable
from typing import Literal
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax.tree_util import Partial

from src.model.aggregation_function import generate_aggregation_function
from src.model.message_function import update_edges, rotate
from src.model.scoring import generate_scoring_activation_function
from util.config.config import RunConfig
from util.helper_classes.customized_graphs_tuple import BatchDependentData
from util.helper_classes.customized_graphs_tuple import EpochIndependentData


def create_model_partial(
    config: RunConfig,
    epoch_independent_info: EpochIndependentData,
    mode: Literal["train", "validation"],
    n_nodes: int,
) -> Callable:
    """
    Function to create a partial of the model, with all necessary parameters set and ready to go
    :param config: Config object holding all parameters of a given trial
    :param epoch_independent_info: helper object, holds pre-compute information about the graph, and the data
    :param mode: Flag using a literal to distinguish between training and validation -> different scorer nad array shapes
    :param n_nodes: number of nodes in the network
    :return:
        A partial of the model.
    """
    match mode:
        case "train":
            n_negative_examples = config.run.data.negative_sampling.n_negative_samples
            n_validation_repetitions = None
        case _:
            n_negative_examples = (
                config.run.data.negative_sampling.n_negative_samples - 1
            )
            n_validation_repetitions = math.ceil(
                n_nodes / config.run.data.negative_sampling.n_negative_samples
            )


    agg_func = generate_aggregation_function(
        func_name=config.run.training.message_passing.aggregation_function,
        n_nodes=n_nodes - 1,  # -1 to exclude virtual node
        query_embedding_dim=config.run.training.message_passing.query_embedding_dimensionality,
    )

    if config.run.training.message_passing.aggregation_function == "pna":
        up_scaling_after_aggregation_factor = 13
    else:
        up_scaling_after_aggregation_factor = 2
    return Partial(
        GraphModelWithEmbeddings,
        edge_representation_dim=config.run.training.message_passing.edge_representation_dimensionalities,
        query_embedding_dim=config.run.training.message_passing.query_embedding_dimensionality,
        # n_nodes=n_nodes + 1,  # +1 wegen paddings
        n_nodes=n_nodes,  # +1 wegen paddings
        messaging_function=config.run.training.message_passing.messanger_function,
        aggregation_function=agg_func,
        query_dependent_edge_representations=config.run.training.message_passing.query_dependent_edge_representations,
        n_unique_edges=(epoch_independent_info.n_unique_edges - 1) * 2 + 1,
        activation_function_for_scoring=generate_scoring_activation_function(
            config.run.training.scoring.activation_function
        ),
        scoring_layer_dim=config.run.training.scoring.dimensionalities,
        n_edges=epoch_independent_info.n_edges,
        activation_function_for_message_passing=generate_scoring_activation_function(
            config.run.training.message_passing.activation_function
        ),
        layer_normalization_in_message_passing=config.run.training.message_passing.layer_normalization,
        n_negative_samples=n_negative_examples,
        upscaling_after_aggregation_factor=up_scaling_after_aggregation_factor,
        mode=mode,
        n_validation_repetitions=n_validation_repetitions,
        indicator_function_as_bounding=config.run.training.message_passing.indicator_function_as_bounding,
        message_augmentation=config.run.training.message_passing.message_augmentation,
        skip_connection=config.run.training.message_passing.skip_connection,
        score_augment_with_query_embedding=config.run.training.scoring.augment_with_query_embedding,
        scorer_type=config.run.training.scoring.type,
        new_representation_based_only_on_update=config.run.training.message_passing.new_representation_based_only_on_update,
        initial_s_dependent_on_p=config.run.training.message_passing.initial_s_dependent_on_p,
        learned_zero_vector=config.run.training.message_passing.learned_zero_vector,
    )


class Layer(nn.Module):
    """
        One layer of the message passing done by Neural Bellman Ford
    """
    query_embedding_dim: int
    edge_representation_dim: Sequence[int]
    n_nodes: int
    messaging_function: str
    aggregation_function: Callable
    query_dependent_edge_representations: bool
    n_unique_edges: int
    activation_function_for_message_passing: Callable
    layer_normalization_in_message_passing: bool
    message_augmentation: bool
    skip_connection: bool
    new_representation_based_only_on_update: bool

    @nn.compact # cool feature of FLAX. Combines Setup with __call_ method
    def __call__(self, graph_attributes, batch_size, degree_out):
        if self.query_dependent_edge_representations:
            generated_embeddings = nn.Dense( #another cool feature, this works by using shape inference
                self.query_embedding_dim * self.n_unique_edges
            )(graph_attributes.query_representation)
            generated_embeddings = jnp.reshape(
                generated_embeddings,
                (batch_size, self.n_unique_edges, self.query_embedding_dim),
            )
            edge_representations = generated_embeddings[:, graph_attributes.edge_type]
        else:
            embedder = nn.Embed(self.n_unique_edges, self.query_embedding_dim)
            edge_representations = embedder.embedding[graph_attributes.edge_type]

        if self.messaging_function == "transe":
            msg_func = jnp.add
        elif self.messaging_function == "rotate":
            msg_func = rotate
        else:
            msg_func = jnp.multiply
        messages = update_edges(
            edge_representations,
            graph_attributes.node_representations,
            graph_attributes.head,
            msg_func,
            batch_size,
        )
        del edge_representations

        # Message augmentation:
        if self.message_augmentation:
            messages = jnp.concatenate(
                (messages, graph_attributes.bounding_conditions), axis=1
            )
            index = jnp.concatenate(
                (
                    jnp.repeat(
                        graph_attributes.tail[jnp.newaxis, :], batch_size, axis=0
                    ),
                    jnp.repeat(
                        jnp.arange(self.n_nodes)[jnp.newaxis, :], batch_size, axis=0
                    ),
                ),
                axis=1,
            )
        else:
            index = jnp.repeat(
                graph_attributes.tail[jnp.newaxis, :], batch_size, axis=0
            )

        update = self.aggregation_function(messages, index, degree_out)
        update = jnp.transpose(update, (1, 0, 2))

        linear = nn.Dense(self.query_embedding_dim)

        if self.new_representation_based_only_on_update:
            input_in_linear_update_layer = update
        else:
            input_in_linear_update_layer = jnp.concatenate(
                (
                    jnp.transpose(graph_attributes.node_representations, (1, 0, 2))[
                        :-1
                    ],  # because padding
                    update,
                ),
                axis=-1,
            )

        new_node_representation = linear(input_in_linear_update_layer)

        if self.layer_normalization_in_message_passing:
            layer_norm = nn.LayerNorm(epsilon=1e-5)
            new_node_representation = layer_norm(new_node_representation)

        if self.activation_function_for_message_passing:
            new_node_representation = self.activation_function_for_message_passing(
                new_node_representation
            )

        new_node_representation = jnp.transpose(new_node_representation, (1, 0, 2))
        if self.skip_connection:
            new_node_representation = (
                new_node_representation + graph_attributes.node_representations[:, :-1]
            )
        graph_attributes.node_representations = (
            graph_attributes.node_representations.at[:, :-1].set(
                new_node_representation
            )
        )
        del new_node_representation
        return graph_attributes


class BellmanFord(nn.Module):
    query_embedding_dim: int
    edge_representation_dim: Sequence[int]
    n_nodes: int
    messaging_function: str
    aggregation_function: Callable
    query_dependent_edge_representations: bool
    n_unique_edges: int
    activation_function_for_message_passing: Callable
    n_edges: int
    layer_normalization_in_message_passing: bool
    n_negative_samples: int
    upscaling_after_aggregation_factor: int
    indicator_function_as_bounding: bool
    message_augmentation: bool
    skip_connection: bool
    new_representation_based_only_on_update: bool
    initial_s_dependent_on_p: bool
    learned_zero_vector: bool

    @nn.compact
    def __call__(
        self,
        graph_attributes: BatchDependentData,
        s: jnp.ndarray,
        p: jnp.ndarray,
        batch_size: int,
        degree_out: jnp.ndarray,
    ):
        """
        :param graph_attributes:
        :param s:
        :param p:
        :param batch_size:
        :param degree_out:
        :return:
        """
        query_embeder = nn.Embed(
            # self.n_unique_edges + 1,
            self.n_unique_edges,
            self.query_embedding_dim,
        )
        query_embeddings = query_embeder(p)
        # query_embeddings = jnp.round(query_embeddings, 6)
        # jax.debug.print("qe: {}", query_embeddings, ordered=True)

        if (not self.indicator_function_as_bounding) or (
            not self.initial_s_dependent_on_p
        ):
            entity_embedder = nn.Embed(
                # self.n_unique_edges + 1,
                self.n_nodes,
                self.query_embedding_dim,
            )

        if self.indicator_function_as_bounding:
            if self.learned_zero_vector:
                bounding_initialized = jnp.zeros(
                    (
                        batch_size,
                        self.n_nodes,
                        1,
                    ),
                    jnp.int8,
                )
                bounding_initialized = nn.Embed(1, self.query_embedding_dim)(
                    bounding_initialized
                ).squeeze()
            else:
                bounding_initialized = jnp.zeros(
                    (
                        batch_size,
                        self.n_nodes,
                        self.query_embedding_dim,
                    )
                )
        else:
            bounding_initialized = entity_embedder(
                jnp.repeat(jnp.arange(self.n_nodes)[None, :], batch_size, axis=0)
            )

        if self.initial_s_dependent_on_p:
            bounding_initialized = bounding_initialized.at[
                jnp.arange(batch_size), s
            ].set(query_embeddings)
        else:
            bounding_initialized.at[jnp.arange(batch_size), s].set(entity_embedder(s))

        # del bounding_zeros
        graph_attributes.bounding_conditions = bounding_initialized
        graph_attributes.query_representation = query_embeddings
        graph_attributes.node_representations = bounding_initialized

        # del bounding_initialized
        for _ in self.edge_representation_dim:
            graph_attributes = nn.remat(Layer, static_argnums=(1, 2))(
                query_embedding_dim=self.query_embedding_dim,
                edge_representation_dim=self.edge_representation_dim,
                n_nodes=self.n_nodes,
                messaging_function=self.messaging_function,
                aggregation_function=self.aggregation_function,
                query_dependent_edge_representations=self.query_dependent_edge_representations,
                n_unique_edges=self.n_unique_edges,
                activation_function_for_message_passing=self.activation_function_for_message_passing,
                layer_normalization_in_message_passing=self.layer_normalization_in_message_passing,
                message_augmentation=self.message_augmentation,
                skip_connection=self.skip_connection,
                new_representation_based_only_on_update=self.new_representation_based_only_on_update,
            )(graph_attributes, batch_size, degree_out)
        return graph_attributes, query_embeddings


class Scorer(nn.Module):
    query_embedding_dim: int
    activation_function_for_scoring: Callable
    scoring_layer_dim: Sequence[int]
    n_negative_samples: int
    type_: Literal["o", "so"]
    augment_with_query_embedding: bool

    @nn.compact
    def __call__(
        self,
        graph_attributes: BatchDependentData,
        query_embeddings,
        s: jnp.ndarray,
        o: jnp.ndarray,
        batch_size: int,
        degree_out: jnp.ndarray,
    ):

        object_representations = graph_attributes.node_representations[
            jnp.repeat(jnp.arange(batch_size), self.n_negative_samples + 1),
            o.flatten(),
        ]
        match self.type_:
            case "o":
                scoring_input = object_representations
                multiplication_factor = 1
            case "so":
                source_representations = graph_attributes.node_representations[
                    jnp.repeat(jnp.arange(batch_size), self.n_negative_samples + 1),
                    jnp.repeat(s, self.n_negative_samples + 1, axis=0).flatten(),
                ]
                scoring_input = jnp.concatenate(
                    (source_representations, object_representations), axis=1
                )
                multiplication_factor = 2
            case "distmult":
                source_representations = graph_attributes.node_representations[
                    jnp.repeat(jnp.arange(batch_size), self.n_negative_samples + 1),
                    jnp.repeat(s, self.n_negative_samples + 1, axis=0).flatten(),
                ]
                predicate_representations = jnp.repeat(
                    query_embeddings, self.n_negative_samples + 1, axis=0
                )
                scores = (
                    source_representations
                    * predicate_representations
                    * object_representations
                ).sum(axis=1)
                return scores.reshape(
                    (batch_size, self.n_negative_samples + 1, 1)
                ).squeeze()
            case "transe":
                source_representations = graph_attributes.node_representations[
                    jnp.repeat(jnp.arange(batch_size), self.n_negative_samples + 1),
                    jnp.repeat(s, self.n_negative_samples + 1, axis=0).flatten(),
                ]
                predicate_representations = jnp.repeat(
                    query_embeddings, self.n_negative_samples + 1, axis=0
                )
                scores = -jnp.linalg.norm(
                    source_representations
                    + predicate_representations
                    - object_representations,
                    axis=1,
                    ord=1,
                )
                return scores.reshape(
                    (batch_size, self.n_negative_samples + 1, 1)
                ).squeeze()
            case "rotate":
                source_representations = graph_attributes.node_representations[
                    jnp.repeat(jnp.arange(batch_size), self.n_negative_samples + 1),
                    jnp.repeat(s, self.n_negative_samples + 1, axis=0).flatten(),
                ]
                predicate_representations = jnp.repeat(
                    query_embeddings, self.n_negative_samples + 1, axis=0
                )
                s_re, s_im = jnp.split(source_representations, 2, axis=-1)
                p_re, p_im = jnp.split(predicate_representations, 2, axis=-1)
                o_re, o_im = jnp.split(object_representations, 2, axis=-1)

                product_re = s_re * p_re - s_im * p_im
                product_im = s_re * p_im + s_im * p_re

                return (
                    -jnp.linalg.norm(
                        jnp.concatenate(
                            [product_re - o_re, product_im - o_im], axis=-1
                        ),
                        axis=1,
                        ord=1,
                    )
                    .reshape((batch_size, self.n_negative_samples + 1, 1))
                    .squeeze()
                )
            case _:
                raise ValueError("Wrong type given, choose between o, so, distmult, rotate and transe")

        if self.augment_with_query_embedding:
            query_embeddings_for_concatenation = jnp.repeat(
                query_embeddings, self.n_negative_samples + 1, axis=0
            )
            scoring_input = jnp.concatenate(
                (scoring_input, query_embeddings_for_concatenation), axis=1
            )
            multiplication_factor += 1
            del query_embeddings_for_concatenation

        scoring_input = scoring_input.reshape(
            (
                batch_size,
                self.n_negative_samples + 1,
                self.query_embedding_dim * multiplication_factor,
            )
        )

        for i, n_neuron in enumerate(self.scoring_layer_dim):
            scoring_input = nn.remat(nn.Dense)(n_neuron)(scoring_input)
            if i != len(self.scoring_layer_dim) - 1:
                scoring_input = self.activation_function_for_scoring(scoring_input)

        return scoring_input.squeeze()


class GraphModelWithEmbeddings(nn.Module):
    query_embedding_dim: int
    edge_representation_dim: Sequence[int]
    n_nodes: int
    messaging_function: str
    aggregation_function: Callable
    query_dependent_edge_representations: bool
    n_unique_edges: int
    activation_function_for_scoring: Callable
    activation_function_for_message_passing: Callable
    scoring_layer_dim: Sequence[int]
    n_edges: int
    layer_normalization_in_message_passing: bool
    n_negative_samples: int
    upscaling_after_aggregation_factor: int
    mode: str
    n_validation_repetitions: int | None
    indicator_function_as_bounding: bool
    message_augmentation: bool
    skip_connection: bool
    score_augment_with_query_embedding: Literal["o", "so"]
    scorer_type: bool
    new_representation_based_only_on_update: bool
    initial_s_dependent_on_p: bool
    learned_zero_vector: bool

    @nn.compact
    def __call__(
        self,
        graph_attributes: BatchDependentData,
        s: jnp.ndarray,
        p: jnp.ndarray,
        o: jnp.ndarray,
        batch_size: int,
        degree_out: jnp.ndarray,
    ):
        graph_attributes, query_embeddings = BellmanFord(
            query_embedding_dim=self.query_embedding_dim,
            edge_representation_dim=self.edge_representation_dim,
            n_nodes=self.n_nodes,
            messaging_function=self.messaging_function,
            aggregation_function=self.aggregation_function,
            query_dependent_edge_representations=self.query_dependent_edge_representations,
            n_unique_edges=self.n_unique_edges,
            activation_function_for_message_passing=self.activation_function_for_message_passing,
            n_edges=self.n_edges,
            layer_normalization_in_message_passing=self.layer_normalization_in_message_passing,
            n_negative_samples=self.n_negative_samples,
            upscaling_after_aggregation_factor=self.upscaling_after_aggregation_factor,
            indicator_function_as_bounding=self.indicator_function_as_bounding,
            message_augmentation=self.message_augmentation,
            skip_connection=self.skip_connection,
            new_representation_based_only_on_update=self.new_representation_based_only_on_update,
            initial_s_dependent_on_p=self.initial_s_dependent_on_p,
            learned_zero_vector=self.learned_zero_vector,
        )(graph_attributes, s, p, batch_size, degree_out)
        if self.mode == "train":
            return Scorer(
                query_embedding_dim=self.query_embedding_dim,
                activation_function_for_scoring=self.activation_function_for_scoring,
                scoring_layer_dim=self.scoring_layer_dim,
                n_negative_samples=self.n_negative_samples,
                type_=self.scorer_type,
                augment_with_query_embedding=self.score_augment_with_query_embedding,
            )(
                graph_attributes=graph_attributes,
                query_embeddings=query_embeddings,
                s=s,
                o=o,
                batch_size=batch_size,
                degree_out=degree_out,
            )
        else:
            output = []
            scorer = Scorer(
                query_embedding_dim=self.query_embedding_dim,
                activation_function_for_scoring=self.activation_function_for_scoring,
                scoring_layer_dim=self.scoring_layer_dim,
                n_negative_samples=self.n_negative_samples,
                augment_with_query_embedding=self.score_augment_with_query_embedding,
                type_=self.scorer_type,
            )
            counter = 0
            for _ in range(self.n_validation_repetitions):
                o_part = o[counter : (counter + self.n_negative_samples + 1)]
                counter = counter + self.n_negative_samples + 1
                o_part = jnp.repeat(o_part[None, :], batch_size, axis=0)
                output.append(
                    scorer(
                        graph_attributes=graph_attributes,
                        query_embeddings=query_embeddings,
                        s=s,
                        o=o_part,
                        batch_size=batch_size,
                        degree_out=degree_out,
                    )[
                        jnp.newaxis,
                        :,
                        :,
                    ]
                )

            return output
