import os
import random
import time
from collections import defaultdict

import jax.random
from flax import serialization

from data.loader import load_and_prepare_data
from src.training_loop.generate_train_test_valid import generate_train_valid_test
from util.config.jax_config import configure_jax_environment
from util.db.connector import Connector


def single_run(input_):
    """
    Runs the complete model training pipeline
    :param input_: a tuple containing a que of possible gpu ids, and another tuple composed of the config for this run
        the number of this run, the logger, and the search id
    """
    q = input_[0]
    gpu_id = q.get()
    run_config, run_number, logger, search_id = input_[1]
    if run_config.run.save_results:
        logger.info(f"Started Trial {run_number}")
        time.sleep(random.randint(1, 30))
        db_connection = Connector(logger=logger)
        db_connection.write_run(
            config=run_config, trial_number=run_number, search_id=search_id
        )
        import pathlib

        hpo_path = (
            pathlib.Path(__file__).parent
            / "search"
            / "hpo_searches"
            / f"{search_id}"
            / str(run_number)
        )
        os.makedirs(hpo_path)
    configure_jax_environment(memory_fraction=0.96, visible_devices=gpu_id)
    data, epoch_independent_train_info = load_and_prepare_data(run_config)
    # needs to be only called after the prepare_data has been called, otherwise the number of nodes is inaccurate
    seed = run_config.run.seed
    initial_key = jax.random.PRNGKey(seed)
    training_key, initialization_key = jax.random.split(initial_key, 2)
    model_steps, train_state = generate_train_valid_test(
        config=run_config,
        data=data,
        epoch_independent_train_info=epoch_independent_train_info,
        initialization_key=initialization_key,
        max_node_in_test=epoch_independent_train_info.max_node,
    )

    best_parameters = {"epoch": 0, "mrr": 0}
    for epoch in range(1, run_config.run.training.n_epochs + 1):
        if run_config.run.save_results:
            logger.info(f"Trial{run_number}: Starting epoch{epoch}")
        time1 = time.time()
        (
            train_state,
            train_metrics,
            training_key,
        ) = model_steps.train_one_epoch(state=train_state, rng_key=training_key)
        time2 = time.time()
        if run_config.run.save_results:
            logger.info(f"Trial{run_number}: took {int(time2 - time1)} seconds")
            logger.info(f"Trial{run_number}: loss: {train_metrics}")
            logger.info(f"Trial{run_number}: started validation")
        else:
            print(f"Trial{run_number}: took {int(time2 - time1)} seconds")
            print(f"Trial{run_number}: loss: {train_metrics}")
            print(f"Trial{run_number}: started validation")

        metrics = model_steps.validate(param=train_state.params)

        time3 = time.time()
        if run_config.run.save_results:
            logger.info(
                f"Trial{run_number}: validating epoch {epoch} took {int(time3 - time2)} seconds"
            )
        else:
            print(f"Trial{run_number}: started validation")
        hits_dict = defaultdict(float)
        for i, value in enumerate(run_config.run.evaluation.hits_at_N):
            hits_dict[f"hits_at_{value}"] = float(metrics[2 + i])
        if run_config.run.save_results:
            logger.info(f"validating Epoch: {epoch} took {int(time3 - time2)} seconds")
            logger.info(
                f"Trial{run_number}: validating epoch {epoch} took {int(time3 - time2)} seconds"
            )
            logger.info(
                f"Trial{run_number}: Validation results obtained in epoch {epoch} \n \t mr: {float(metrics[0])}  \n\t mrr: {float(metrics[1])}  \n\t {dict(hits_dict)}"
            )
            db_connection.write_eval(
                run_id=run_number,
                search_id=search_id,
                metrics=metrics,
                epoch=epoch,
                hits_at_N=hits_dict,
            )
            if metrics[1] > best_parameters["mrr"]:
                with open(hpo_path / f"params.bytes", "wb") as out:
                    out.write(serialization.to_bytes(train_state.params))
                best_parameters["mrr"] = metrics[1]
                best_parameters["epoch"] = epoch
        else:
            print(
                f"Trial{run_number}: Validation results obtained in epoch {epoch} \n \t mr: {float(metrics[0])}  \n\t mrr: {float(metrics[1])}  \n\t {dict(hits_dict)}"
            )

    if run_config.run.save_results:
        best_epoch, mrr = db_connection.read_best(
            search_id=search_id, run_id=run_number
        )
        logger.info(
            f"Run {run_number}: Best validation mrr {mrr} in epoch {best_epoch}"
        )
        with open(hpo_path / f"params.bytes", "rb") as best_p:
            best_params = serialization.from_bytes(train_state.params, best_p.read())
        test_metrics = model_steps.test(best_params)
        db_connection.write_test(
            run_id=run_number,
            search_id=search_id,
            metrics=test_metrics,
            hits_at_N=run_config.run.evaluation.hits_at_N,
            training_epoch=best_epoch,
        )
        logger.info(f"Run {run_number}: Test mrr = {test_metrics[0]}")
    q.put(gpu_id)
    return None
