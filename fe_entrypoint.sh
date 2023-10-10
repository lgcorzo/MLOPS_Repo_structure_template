#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh
# init conda
conda init
# activate the environment
conda activate code_development_env

echo "Starting frontend..."
gunicorn Code.FrontEnd.app:server --workers 4 --threads 2 --bind $FE_HOST:$FE_PORT