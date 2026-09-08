#!/bin/bash
# Environment and pipeline configuration for running GenEval with run_diffusers.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# GenEval prompt files
PROMPTS_FILE="filtered_prompts.json"
METADATA_FILE="evaluation_metadata.jsonl"

# TIR pipeline settings
MODEL="sdxl"                                       # sdxl, sd3, or sana1.5
MLLM="qwen"                                        # qwen or gpt4o
REFINEMENT_ITERATIONS=3
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/geneval_batch_results_diffusers"
GENEVAL_OUTPUT_DIR="gen_images"