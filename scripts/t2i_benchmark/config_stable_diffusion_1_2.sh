#!/bin/bash
# Environment and pipeline configuration for the custom T2I benchmark run

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/evaluation

# T2I benchmark files
DATASET_FILE="t2i_benchmark_dataset.json"
QUERY_JSON_DIR="evaluation/json_files"

# TIR pipeline settings
MODEL_VERSION="1.5"
MLLM="qwen"                                        # qwen or gpt4o
RESTART_STEPS=(0 0 0)
REFINEMENT_STEP=99
SELECTION_MODE="first_early_stop"                  # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/t2i_benchmark_results"
RESULTS_DIR="t2i_benchmark_results"