#!/bin/bash
# Environment and pipeline configuration for running GenEval with run_stable_diffusion_1_2.py

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# GenEval prompt files
PROMPTS_FILE="filtered_prompts.json"
METADATA_FILE="evaluation_metadata.jsonl"          # GenEval's 553 prompts (single_object, two_object, counting, colors, position, color_attr)

# TIR pipeline settings
MODEL_VERSION="1.4"                                # Stable Diffusion version: 1.4, 1.5, or 2.1
MLLM="qwen"                                        # Prompt judge/refiner: qwen (local Qwen2.5-VL) or gpt4o (Azure OpenAI, needs ENDPOINT_URL/AOAI_JUDGE_MODEL env vars + az login)
RESTART_STEPS=(0 0 0)                              # One entry per refinement round; 0 = full restart from noise
REFINEMENT_STEP=99                                 # Denoising step at which the MLLM judges the image (out of 100)
SELECTION_MODE="final_only"                        # Which image counts as "final" per prompt when post-processing: first_early_stop (ES1->ES2->ES3->last round) or final_only (always the last round)

# Output locations
OUTPUT_DIR="new_outputs/geneval_batch_results_SD"  # Where main.py writes raw per-prompt outputs
GENEVAL_OUTPUT_DIR="gen_images"                    # Where post_process_geneval.py writes GenEval-ready output