# Get the actual path
current_path=$(pwd)
echo "Actual path: $current_path"
#!/bin/bash

# Navigate up one level in the directory tree
cd ..
cd ..
root_path=$(pwd)
echo "root path: $root_path"

export PYTHONPATH=$root_path

# Check if conda is installed and in the path
if ! command -v conda &> /dev/null
then
    echo "conda is not installed or not in the path"
    exit 1
fi

# Activate the conda environment spark_env
echo "Activating conda environment code_development_env"
source activate code_development_env

# Check if the activation was successful
if [ $? -eq 0 ]
then
    echo "Conda environment code_development_env activated"
else
    echo "Failed to activate conda environment code_development_env"
    exit 1
fi

cd $root_path
echo "Start FrontEnd"
python -m  Code.FrontEnd.app
