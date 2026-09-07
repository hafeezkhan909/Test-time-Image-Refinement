#!/bin/bash
# Runs run_diffusers.py (sdxl/sd3/sana1.5) against the GenEval benchmark prompts,
# then reorganizes the output into the folder structure GenEval expects.

set -e
source scripts/geneval/config_diffusers.sh

# Step 1: Extract GenEval prompts, grouped by tag
if [ -f "$PROMPTS_FILE" ]; then
    echo "$PROMPTS_FILE already exists. Skipping extraction."
else
    echo "Extracting prompts from $METADATA_FILE..."
    python scripts/extract/extract_prompts.py
fi

# Step 2: Run the TIR refinement loop on the GenEval prompts
echo "Running the diffusion pipeline ($MODEL) on GenEval prompts..."
python src/run_diffusers.py \
    --model "$MODEL" \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --mllm "$MLLM"

# Step 3: Reorganize output into GenEval's expected structure
echo "Post-processing images into GenEval format..."
python scripts/post_process/post_process_geneval.py \
    --metadata_file "$METADATA_FILE" \
    --output_dir "$GENEVAL_OUTPUT_DIR" \
    --generated_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo ""
echo "Done. GenEval-ready images are in $GENEVAL_OUTPUT_DIR."