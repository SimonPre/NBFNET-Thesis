import os

import jax


def configure_jax_environment(
    memory_fraction: float | None, visible_devices: str | None, preallocate: bool = True
):
    """
        Helper function to manage the gpu usage of jax
        :param memory_fraction: fraction of GPU memory to be pre-allocated by jax
        :param visible_devices: provide a string of form "0,1" or just "1" to limit jax's access to other gpus
        :param preallocate: Flag if the memory of the gpu should be pre-allocated, defaults to True because this is
            leads to more efficient memory usage
    """
    if preallocate:
        if memory_fraction:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if visible_devices:
        jax.config.values["jax_cuda_visible_devices"] = visible_devices
