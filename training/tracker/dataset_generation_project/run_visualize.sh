#!/bin/bash
# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Execute python script in visualization mode
python3 "${SCRIPT_DIR}/dataset_generator_from_video.py" "${SCRIPT_DIR}/videos4dataset" --visualize
