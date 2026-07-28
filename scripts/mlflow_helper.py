from __future__ import annotations

from contextlib import contextmanager

import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "clash_royale_winner_prediction"


def init_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@contextmanager
def stage_run(run_name: str, tags: dict | None = None):
    init_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run
