#!/bin/bash
# Runs run_dalle.py against the DrawBench prompts, then flattens the output
# into a single folder of prompt_XXX.png images for scoring separately.
#
# Requires Azure OpenAI credentials set separately (not in this file):
#   export ENDPOINT_URL="https://<your-resource>.openai.azure.com/"
#   export DALLE_MODEL="dalle3"
#   export AOAI_JUDGE_MODEL="gpt-4o"   # only needed if MLLM=gpt4o below
#   az login

set -e
source scripts/drawbench/config_dalle.sh

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
echo "Running the DALL-E pipeline on DrawBench prompts..."
python src/run_dalle.py \
    --prompts_file "$DRAWBENCH_PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --size "$SIZE" \
    --quality "$QUALITY" \
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