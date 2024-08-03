import datetime
import multiprocessing
import os
import pathlib
from multiprocessing import Pool

import jax
import jax.random
import jax.random
import yaml
from omegaconf import OmegaConf
from yaml.loader import SafeLoader

from main import single_run
from util.config.config import creat_config
from util.db.connector import Connector
from util.logger import create_logger


class Search:
    def __init__(self, n_trials: int, search_type: str, devices: str):
        jax.config.values["jax_cuda_visible_devices"] = devices
        self.n_trials = n_trials
        self.search_type = search_type
        self.n_devices = len(jax.devices())
        logg_path = pathlib.Path(__file__).parent / "logs"
        self.date = datetime.date
        self.logger = create_logger(
            f"search_{datetime.datetime.now().date()}", logdir=logg_path
        )
        self.logger.info(f"Started Search")
        self.devices = devices
        self.logger.info(f"Current Date{datetime.datetime.now()}")
        self.logger.info(
            f"Running {n_trials} {search_type} trials on devices {devices}"
        )
        self.db_connector = Connector(self.logger)
        self.search_id = self.db_connector.write_search(n_trials=n_trials)
        hpo_path = pathlib.Path(__file__).parent / "hpo_searches"
        if not hpo_path.exists():
            os.makedirs(hpo_path)
        path_to_search_parameters = hpo_path / f"{self.search_id}"
        os.makedirs(path_to_search_parameters)
        self.counter = 0

    def search(self, debug: bool):
        if debug:
            configs = [
                creat_config() for user_config in self._generate_user_config_no_change()
            ]
        else:
            if self.search_type == "ablation":
                configs = [
                    creat_config(user_config)
                    for user_config in self._generate_user_config_ablation()
                ]
            else:
                configs = [
                    creat_config(user_config)
                    for user_config in self._generate_user_config()
                ]
        run_number = [i for i in range(self.n_trials)]
        logger = [self.logger for _ in range(self.n_trials)]
        search_id = [self.search_id for _ in range(self.n_trials)]

        multiprocessing.set_start_method("spawn")
        with Pool(self.n_devices, maxtasksperchild=1) as p:
            manager = multiprocessing.Manager()
            passable_queue1 = manager.Queue()
            input_ = [list(x) for x in zip(configs, run_number, logger, search_id)]
            for device in self.devices.split(","):
                passable_queue1.put(device)
            p.map(single_run, ((passable_queue1, in_) for in_ in input_), chunksize=1)

    def _generate_user_config(self):
        "Generates configs based on the configs based on libkge folders"
        search_space_location = (
            pathlib.Path(__file__).parent / "search" / "search_space"
        )
        config_folders = [x[0] for x in os.walk(search_space_location)]
        output_ = []
        for folder in config_folders[1:]:
            with open(folder + "/config.yaml", "rb") as config_file:
                loaded = yaml.load(config_file, SafeLoader)
            relevant = loaded["user"]
            relevant["gnn_dim"] = 32
            config_dict = {
                "run": {
                    "data": {
                        "negative_sampling": {
                            "n_negative_samples": relevant["negative_samples"]
                        },
                    },
                    "training": {
                        "batch_size": relevant["batch_size"],
                        "adversarial_temperature": relevant["temperature"],
                        "optimizer": {"learning_rate": relevant["learning_rate"]},
                    },
                }
            }
            output_.append(config_dict)
        return output_

    def _generate_user_config_no_change(self):
        "Helper mostly used for debugging"
        return [0, 1, 2, 3]

    def _generate_user_config_ablation(self):
        "Generates configs based on the configs contained in the ablation folder"
        ablation_configs_path = (
            pathlib.Path(__file__).parent / "search" / "search_space" / "ablation"
        )
        output = []
        config_files = [x[2] for x in os.walk(ablation_configs_path)]
        for file in config_files[0]:
            output.append(OmegaConf.load(ablation_configs_path / file))
        return output


if __name__ == "__main__":
    Search(n_trials=1, search_type="search", devices="0").search(debug=True)
