from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.tree_util import Partial
from optax import sigmoid_binary_cross_entropy

from util.config.conf_dataclass import RunConfig


def generate_train_step_function(
    config: RunConfig,
    model_generator: Callable,
    n_negative_samples: int,
):
    """
    used in train_on_epoch. Computes gradients for singe mini-batch
    :param config: the config object holding all parameters
    :param model_generator: Callable with the ability to generate a model instance
    :param n_negative_samples: number of negative examples to create
    :return: loss and more importantly the gradient for a given min-batch
    """
    static_jit = Partial(jax.jit, static_argnames=["batch_size"])

    @static_jit
    def train_step(state, graph, s, p, o, batch_size, degree_out):
        def loss_fn(params):
            logits = (
                model_generator()
                .apply(params, graph, s, p, o, batch_size, degree_out)
                .squeeze()
            )
            labels = jnp.concatenate(
                (
                    jnp.ones((batch_size,))[:, jnp.newaxis],
                    jnp.zeros((batch_size, n_negative_samples)),
                ),
                axis=1,
            ).squeeze()
            loss_ = sigmoid_binary_cross_entropy(logits, labels)

            neg_weight = jnp.ones((batch_size, n_negative_samples + 1))
            if config.run.training.adversarial_temperature > 0:
                neg_weight = jax.lax.stop_gradient(
                    neg_weight.at[:, 1:].set(
                        nn.softmax(
                            logits[:, 1:] / config.run.training.adversarial_temperature,
                            axis=-1,
                        )
                    )
                )
            else:
                neg_weight = neg_weight.at[:, 1:].set(1 / n_negative_samples)

            batch_loss = jnp.sum(loss_ * neg_weight, axis=-1) / jnp.sum(
                neg_weight, axis=-1
            )
            return jnp.mean(batch_loss)

        return jax.value_and_grad(loss_fn)(state.params)

    return train_step
