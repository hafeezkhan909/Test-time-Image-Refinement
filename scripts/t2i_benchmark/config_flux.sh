#!/bin/bash
# Environment and pipeline configuration for running the T2I benchmark with run_flux.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/evaluation

# T2I benchmark files
DATASET_FILE="t2i_benchmark_dataset.json"
QUERY_JSON_DIR="evaluation/json_files"

# TIR pipeline settings
MLLM="qwen"                                        # qwen or gpt4o
REFINEMENT_ITERATIONS=3
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/t2i_benchmark_results_flux"
RESULTS_DIR="t2i_benchmark_results_flux"