#!/bin/bash
# Environment and pipeline configuration for the DrawBench run

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# DrawBench prompt files
DRAWBENCH_PROMPTS_FILE="drawbench_prompts.json"
IMAGE_PROMPT_MAPPING_FILE="image_prompt_mapping.json"

# TIR pipeline settings
MODEL_VERSION="1.4"
MLLM="qwen"                                        # qwen or gpt4o
RESTART_STEPS=(0 0 0)
REFINEMENT_STEP=99
SELECTION_MODE="final_only"                        # first_early_stop or final_only

# Output locations
OUTPUT_DIR="new_outputs/drawbench_batch_results_SD"
DRAWBENCH_OUTPUT_DIR="drawbench_images"