#!/bin/bash
# Environment and pipeline configuration for running DrawBench with run_dalle.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# DrawBench prompt files
DRAWBENCH_PROMPTS_FILE="drawbench_prompts.json"
IMAGE_PROMPT_MAPPING_FILE="image_prompt_mapping.json"

# TIR pipeline settings
MLLM="qwen"                                        # qwen or gpt4o
REFINEMENT_ITERATIONS=3
SIZE="1024x1024"                                   # 1024x1024, 1792x1024, or 1024x1792
QUALITY="standard"                                 # standard or hd
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/drawbench_batch_results_dalle"
DRAWBENCH_OUTPUT_DIR="drawbench_images"