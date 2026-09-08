#!/bin/bash
# Environment and pipeline configuration for running GenEval with run_dalle.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# GenEval prompt files
PROMPTS_FILE="filtered_prompts.json"
METADATA_FILE="evaluation_metadata.jsonl"

# TIR pipeline settings
MLLM="qwen"                                        # qwen or gpt4o
REFINEMENT_ITERATIONS=3
SIZE="1024x1024"                                   # 1024x1024, 1792x1024, or 1024x1792
QUALITY="standard"                                 # standard or hd
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/geneval_batch_results_dalle"
GENEVAL_OUTPUT_DIR="gen_images"