import datetime
import json
import pathlib
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from logging import Logger
from uuid import uuid4 as uuid

import firebase_admin
from dotenv import dotenv_values
from firebase_admin import credentials, firestore
from jax import numpy as jnp

from util.config.config import RunConfig


def root_directory():
    return str(pathlib.Path(__file__).parent.parent.parent)


class Connector:
    RUN_COLLECTION = "Runs"
    EVAL_COLLECTION = "Evaluation"
    TEST_COLLECTION = "Test"
    SEARCH_COLLECTION = "Search"
    GPU_MANAGER = "GPU"

    """ Helper Class to connect to a Firebase DB to store the results"""
    def __init__(self, logger: Logger):
        root = root_directory()
        connector_config = dotenv_values(dotenv_path=root + "/util/db/.env")
        cred = credentials.Certificate(json.loads(connector_config["key"]))
        self.run_id = str(uuid())
        admin_app = firebase_admin.initialize_app(credential=cred, name=self.run_id)
        self.db = firestore.client(admin_app)
        self.logger = logger

    def write_run(self, trial_number: int, config: RunConfig, search_id):
        self.logger.info(f"Added run {self.run_id} to search in db")
        self.db.collection(self.SEARCH_COLLECTION).document(search_id).collection(
            self.RUN_COLLECTION
        ).add(
            {
                "trial_number": trial_number,
                "run_id": self.run_id,
                "start_date": datetime.datetime.now(),
                "config": asdict(config),
            }
        )

    def write_search(self, n_trials, **kwargs):
        document_reference = self.db.collection(self.SEARCH_COLLECTION).add(
            {
                "search_date": datetime.datetime.now(),
                "n_trials": n_trials,
                "misc": kwargs,
            }
        )[1]
        self.logger.info(f"Added run {self.run_id} to search in db")
        return document_reference.id

    def write_eval(
        self,
        run_id: int,
        search_id: str,
        metrics: jnp.ndarray,
        epoch: int,
        hits_at_N: dict[str:float],
    ):
        self.db.collection(self.SEARCH_COLLECTION).document(search_id).collection(
            self.EVAL_COLLECTION
        ).add(
            {
                "run_id": run_id,
                "mr": float(metrics[0]),
                "mrr": float(metrics[1]),
                "hits_at_N": hits_at_N,
                "epoch": epoch,
            }
        )
        self.logger.info(f"Run: {run_id}: Epoch {epoch} added validation metrics to db")

    def write_test(
        self,
        run_id: int,
        search_id: str,
        metrics: jnp.ndarray,
        hits_at_N: Sequence[int | None],
        training_epoch: int,
    ):
        hits_dict = defaultdict(float)
        for i, value in enumerate(hits_at_N):
            hits_dict[f"hits_at_{value}"] = float(metrics[2 + i])

        self.db.collection(self.SEARCH_COLLECTION).document(search_id).collection(
            self.TEST_COLLECTION
        ).add(
            {
                "run_id": run_id,
                "mr": float(metrics[0]),
                "mrr": float(metrics[1]),
                "hits_at_N": hits_dict,
                "epoch": training_epoch,
            }
        )
        self.logger.info(
            f"Run: {run_id}: Epoch {training_epoch} added test metrics to db"
        )

    def read_best(self, search_id: str, run_id: int):
        query_result = (
            self.db.collection(self.SEARCH_COLLECTION)
            .document(search_id)
            .collection(self.EVAL_COLLECTION)
            .order_by("mrr", direction=firestore.Query.DESCENDING)
            .where("run_id", "==", run_id)
            .limit(1)
            .get()
        )
        epoch = query_result[0].to_dict()["epoch"]
        mrr = query_result[0].to_dict()["mrr"]
        return epoch, mrr


if __name__ == "__main__":
    connector = Connector()
    test = json.load(open("/home/simon/Downloads/thesis-77afd-33a6631a3004.json"))
