#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh
# init conda
conda init
# activate the environment
conda activate code_development_env

echo "Starting mlflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host '127.0.0.1' -p 1234 &

echo "Starting backend..."
gunicorn Code.Controller.app:app --workers 4 --threads 2 --bind $BE_HOST:$BE_PORT


