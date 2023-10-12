#!/bin/bash

# enable conda for this shell
. /opt/conda/etc/profile.d/conda.sh
# init conda
conda init
# activate the environment
conda activate code_development_env

echo "Starting frontend..."
uvicorn Code.FrontEnd.app:app --host $FE_HOST --port $FE_PORT --workers 1 --proxy-headers