#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh
# init conda
conda init
# activate the environment
conda activate code_development_env

echo "Starting mlflow..."
mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --default-artifact-root ./artifacts --host $BE_HOST -p 1234 &

echo "Starting backend..."
uvicorn Code.Controller.app:app --host $BE_HOST --port $BE_PORT --workers 4 --proxy-headers


