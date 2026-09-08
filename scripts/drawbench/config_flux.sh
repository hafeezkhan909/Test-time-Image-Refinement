#!/bin/bash
# Environment and pipeline configuration for running DrawBench with run_flux.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# DrawBench prompt files
DRAWBENCH_PROMPTS_FILE="drawbench_prompts.json"
IMAGE_PROMPT_MAPPING_FILE="image_prompt_mapping.json"

# TIR pipeline settings
MLLM="qwen"                                        # qwen or gpt4o
REFINEMENT_ITERATIONS=3
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/drawbench_batch_results_flux"
DRAWBENCH_OUTPUT_DIR="drawbench_images"