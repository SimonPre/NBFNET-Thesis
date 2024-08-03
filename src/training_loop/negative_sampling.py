import pathlib
import pickle
from collections import defaultdict
from collections.abc import Callable
from typing import Literal
from typing import Sequence

import jax
import jax.experimental.sparse as sparse
import jax.numpy as jnp
import numpy as np
from jax import random


def subject_object_inversion(
    s: jnp.ndarray,
    p: jnp.ndarray,
    o: jnp.ndarray,
    sample_space: Sequence[int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Helper to invert a set of triples
    :param s: the subjects  indices
    :param p: the predicates indices
    :param o: the object indices
    :param sample_space: [highest node index, highest edge index, highest node index]
    :return:
    """
    upper_s, lower_s = jnp.split(s, 2, axis=0)
    upper_p, lower_p = jnp.split(p, 2, axis=0)
    upper_o, lower_o = jnp.split(o, 2, axis=0)
    lower_p = lower_p + sample_space[1]
    s_new = jnp.concatenate((upper_s, lower_o))
    p_new = jnp.concatenate((upper_p, lower_p))
    o_new = jnp.concatenate((upper_o, lower_s))
    return s_new, p_new, o_new


def parring_function(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Comment 2 authored by BlueRaja under the answer given by nawfal

    https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
    """
    return (x << 16) + y


def generate_negative_sampling_function(
    sample_space: Sequence[int],
    n_samples: int,
    batch_size: int,
    existing_triples: jnp.ndarray | None,
    mode: Literal["random", "random_filtered", "all"],
    dataset_name: str,
) -> Callable:
    """
    Helper function, that returns a partial of the negative sampling function, for easier jit compilation
    :param sample_space: Maximum number of s,p, and o in the graph_attributes. Defines the space from which
     samples can be drawn.
    :param n_samples: Number of permutations drawn for each s,p,o individually: e.g. 0,1,2 means that s stays constant
        p is changed once randomly, and o is changed two times. Adding three false negatives in total.
    :param batch_size: The size of each batch before drawing the negative samples.
    :param existing_triples: the kg, as set of triples
    :param mode: flag to distinguish between random, random filtered and all. all is used in metric computation and
        produces all possible o as negatives.
    :param dataset_name: name of the dataset.
    :return: A partial of the negative sampling function.
    """

    if mode == "random_filtered":
        path_non_acceptable_s = (
            pathlib.Path(__file__).parent.parent.parent
            / f"data/datasets/{dataset_name}/non_acceptable_s.bytes"
        )
        path_non_acceptable_o = (
            pathlib.Path(__file__).parent.parent.parent
            / f"data/datasets/{dataset_name}/non_acceptable_o.bytes"
        )

        # written to disc because this can take quit some time
        if path_non_acceptable_s.exists():
            with open(str(path_non_acceptable_s), "rb") as data_file:
                byte_data = data_file.read()
                non_acceptable_s = pickle.loads(byte_data)
            with open(str(path_non_acceptable_o), "rb") as data_file:
                byte_data = data_file.read()
                non_acceptable_o = pickle.loads(byte_data)
        else:
            hash_sp = parring_function(existing_triples[:, 0], existing_triples[:, 1])
            hash_po = parring_function(existing_triples[:, 1], existing_triples[:, 2])

            non_acceptable_o = defaultdict(lambda: jnp.array([]))
            for entry in np.unique(np.array(hash_sp)):
                filter_ = jnp.where(hash_sp == entry)
                # conversion done because pickling jnp.arrays does currently not respect the device context manager
                non_acceptable_o[entry] = np.unique(
                    np.array(existing_triples[filter_, 2].squeeze())
                )

            non_acceptable_s = defaultdict(lambda: jnp.array([]))
            for entry in np.unique(np.array(hash_po)):
                filter_ = jnp.where(hash_po == entry)
                non_acceptable_s[entry] = np.unique(
                    np.array(existing_triples[filter_, 0].squeeze())
                )

            shape_non_acceptable_s = sum([len(v) for v in non_acceptable_s.values()])
            shape_non_acceptable_o = sum([len(v) for v in non_acceptable_o.values()])

            data_non_acceptable_s = jnp.ones((shape_non_acceptable_s))
            data_non_acceptable_o = jnp.ones((shape_non_acceptable_o))

            non_acceptable_o_index = []
            for k, v in non_acceptable_o.items():
                new = jnp.stack(jnp.meshgrid(jnp.array([k]), v), axis=-1).squeeze()
                if len(new.shape) < 2:
                    non_acceptable_o_index.append(new[jnp.newaxis, :])
                else:
                    non_acceptable_o_index.append(new)

            non_acceptable_o_index = jnp.concatenate(non_acceptable_o_index, axis=0)

            non_acceptable_s_index = []
            for k, v in non_acceptable_s.items():
                new = jnp.stack(jnp.meshgrid(jnp.array([k]), v), axis=-1).squeeze()
                if len(new.shape) < 2:
                    non_acceptable_s_index.append(new[jnp.newaxis, :])
                else:
                    non_acceptable_s_index.append(new)

            non_acceptable_s_index = jnp.concatenate(non_acceptable_s_index, axis=0)

            n_real_nodes = max(sample_space[0], sample_space[2])

            non_acceptable_s = sparse.BCOO(
                (data_non_acceptable_s.flatten(), non_acceptable_s_index),
                shape=(max([k for k in non_acceptable_s.keys()]), n_real_nodes),
            )

            non_acceptable_o = sparse.BCOO(
                (data_non_acceptable_o.flatten(), non_acceptable_o_index),
                shape=(max([k for k in non_acceptable_o.keys()]), n_real_nodes),
            )

            with open(str(path_non_acceptable_s), "wb") as outfile:
                packed = pickle.dumps(non_acceptable_s)
                outfile.write(packed)

            with open(str(path_non_acceptable_o), "wb") as outfile:
                packed = pickle.dumps(non_acceptable_o)
                outfile.write(packed)


    @jax.vmap
    def mutate_filtered(key, p_values):
        return jax.random.choice(
            key,
            jnp.arange(max(sample_space[0], sample_space[2])),
            (1, n_samples),
            replace=False,
            p=p_values,
        )

    def create_negative_samples(
        input_triples: jnp.array,
        random_key: jnp.array,
    ) -> jnp.array:
        """
        :param input_triples: The original batch with true positive triples as jnp array [[0,1,0],[1,0,2],...]
        :param random_key: random key used for sampling.
        :return:
            Returns the initial batch of positive examples, to which the negative examples are appended.
        """

        s_full_size = jnp.repeat(
            input_triples[:, 0][:, jnp.newaxis],
            n_samples + 1,
            axis=1,
        )
        original_s, to_mutate_s = jnp.split(s_full_size, 2)

        o_full_size = jnp.repeat(
            input_triples[:, 2][:, jnp.newaxis],
            n_samples + 1,
            axis=1,
        )
        to_mutate_o, original_o = jnp.split(o_full_size, 2)

        match mode:
            case "random":
                mutations_s = random.randint(
                    random_key,
                    # shape=(int(alpha * n),),
                    shape=(int(batch_size / 2), n_samples),
                    minval=0,
                    maxval=sample_space[0],
                    dtype=jnp.uint32,
                )
                mutations_o = random.randint(
                    random_key,
                    # shape=(int(alpha * n),),
                    shape=(int(batch_size / 2), n_samples),
                    minval=0,
                    maxval=sample_space[2],
                    dtype=jnp.uint32,
                )
            case "random_filtered":
                sp, po = jnp.split(input_triples, 2)

                hashs_sp = np.array(parring_function(sp[:, 0], sp[:, 1]))
                hashs_po = np.array(parring_function(po[:, 1], po[:, 2]))
                p_values_o = jnp.ones((int(batch_size / 2), sample_space[2]))
                p_values_s = jnp.ones((int(batch_size / 2), sample_space[0]))

                p_values_o = p_values_o - non_acceptable_o[hashs_sp].todense()
                p_values_s = p_values_s - non_acceptable_s[hashs_po].todense()

                # non_acceptables_s = [non_acceptable_s[hash_] for hash_ in hashs_po]

                # for i in range(int(batch_size / 2)):
                #    if non_acceptables_o[i].size:
                #        p_values_o = p_values_o.at[i, non_acceptables_o[i]].set(0)

                #    if non_acceptables_s[i].size:
                #        p_values_s = p_values_s.at[i, non_acceptables_s[i]].set(0)

                p_values_o = p_values_o / jnp.sum(p_values_o, axis=1)[:, jnp.newaxis]
                p_values_s = p_values_s / jnp.sum(p_values_s, axis=1)[:, jnp.newaxis]

                random_key, *keys = jax.random.split(random_key, batch_size + 1)
                keys_s, keys_o = jnp.split(jnp.array(keys), 2, axis=0)
                mutations_s = mutate_filtered(keys_s, p_values_s).squeeze()
                mutations_o = mutate_filtered(keys_o, p_values_o).squeeze()

            case "all":
                assert n_samples == max(sample_space[0], sample_space[2])
                assert sample_space[0] == sample_space[2]
                mutations_s = jnp.repeat(
                    jnp.arange(sample_space[0])[jnp.newaxis, :],
                    int(batch_size / 2),
                    axis=0,
                )
                mutations_o = jnp.repeat(
                    jnp.arange(sample_space[2])[jnp.newaxis, :],
                    int(batch_size / 2),
                    axis=0,
                )
            case _:
                raise ValueError(
                    f"mode: {mode} is not permissible, use either random, random_filtered, or all"
                )

        mutated_s = to_mutate_s.at[:, 1:].set(mutations_s)
        mutated_o = to_mutate_o.at[:, 1:].set(mutations_o)

        s_new = jnp.concatenate((original_s, mutated_s))
        o_new = jnp.concatenate((mutated_o, original_o))

        # s_new, p_new, o_new = subject_object_inversion(
        #    s=s_new,
        #    p=input_triples[:, 1],
        #    o=o_new,
        #    sample_space=sample_space,
        # )

        return s_new, input_triples[:, 1], o_new

    return create_negative_samples
