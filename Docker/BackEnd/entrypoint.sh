#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh
# init conda
conda init
# activate the environment
conda activate code_development_env

echo "Starting backend..."
gunicorn Code.Controller.app:app --workers 4 --threads 2 --bind 0.0.0.0:8000 


