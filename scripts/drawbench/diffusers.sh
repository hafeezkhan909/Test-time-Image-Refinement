#!/bin/bash
# Runs run_diffusers.py (sdxl/sd3/sana1.5) against the DrawBench prompts, then
# flattens the output into a single folder of prompt_XXX.png images for scoring separately.

set -e
source scripts/drawbench/config_diffusers.sh

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
echo "Running the diffusion pipeline ($MODEL) on DrawBench prompts..."
python src/run_diffusers.py \
    --model "$MODEL" \
    --prompts_file "$DRAWBENCH_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --mllm "$MLLM"

# Step 4: Flatten output into prompt_XXX.png images
echo "Post-processing images..."
python scripts/post_process/post_process_drawbench.py \
    --output_dir "$DRAWBENCH_OUTPUT_DIR" \
    --generated_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo ""
echo "Done. Images are in $DRAWBENCH_OUTPUT_DIR."