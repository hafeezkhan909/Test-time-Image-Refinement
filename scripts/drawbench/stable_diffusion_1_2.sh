#!/bin/bash
# Runs the TIR pipeline against the DrawBench prompts, then flattens the
# output into a single folder of prompt_XXX.png images for scoring separately.

set -e
source scripts/drawbench/config_stable_diffusion_1_2.sh

LAST_RESTART_STEP="${RESTART_STEPS[${#RESTART_STEPS[@]}-1]}"

# Step 1: Pull DrawBench prompts
if [ -f "$DRAWBENCH_PROMPTS_FILE" ]; then
    echo "$DRAWBENCH_PROMPTS_FILE already exists. Skipping."
else
    echo "Fetching DrawBench prompts..."
    python src/prepare_drawbench_prompts.py
fi

# Step 2: Build the image-to-prompt mapping
if [ -f "$IMAGE_PROMPT_MAPPING_FILE" ]; then
    echo "$IMAGE_PROMPT_MAPPING_FILE already exists. Skipping."
else
    echo "Building image-to-prompt mapping..."
    python src/image_prompt_mapping.py
fi

# Step 3: Run the TIR refinement loop
echo "Running the diffusion pipeline (Stable Diffusion $MODEL_VERSION) on DrawBench prompts..."
python src/run_stable_diffusion_1_2.py \
    --model_version "$MODEL_VERSION" \
    --prompts_file "$DRAWBENCH_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --restart_steps "${RESTART_STEPS[@]}" \
    --refinement_step "$REFINEMENT_STEP" \
    --mllm "$MLLM"

# Step 4: Flatten output into prompt_XXX.png images
echo "Post-processing images..."
python scripts/post_process/post_process_drawbench.py \
    --output_dir "$DRAWBENCH_OUTPUT_DIR" \
    --generated_dir "$OUTPUT_DIR" \
    --refinement_iterations "${#RESTART_STEPS[@]}" \
    --last_restart_step "$LAST_RESTART_STEP" \
    --selection_mode "$SELECTION_MODE"

echo ""
echo "Done. Images are in $DRAWBENCH_OUTPUT_DIR."