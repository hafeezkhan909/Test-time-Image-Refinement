#!/bin/bash
# Environment configuration

# Set default paths
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export PROMPTS_FILE="filtered_prompts.json"
export OUTPUT_DIR="new_outputs/batch_results"