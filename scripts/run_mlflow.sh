#!/usr/bin/env bash
MLFLOW_HOME=$HOME/mlflow_data_rec_sys

mkdir -p $MLFLOW_HOME/artifacts

mlflow server \
        --backend-store-uri sqlite:///$MLFLOW_HOME/mlflow.db \
        --default-artifact-root file://$MLFLOW_HOME/artifacts \
        --host 127.0.0.1 \
        --port 5000
