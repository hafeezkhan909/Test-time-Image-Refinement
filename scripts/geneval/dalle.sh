#!/bin/bash
# Runs run_dalle.py against the GenEval benchmark prompts, then reorganizes
# the output into the folder structure GenEval expects.
#
# Requires Azure OpenAI credentials set separately (not in this file):
#   export ENDPOINT_URL="https://<your-resource>.openai.azure.com/"
#   export DALLE_MODEL="dalle3"
#   export AOAI_JUDGE_MODEL="gpt-4o"   # only needed if MLLM=gpt4o below
#   az login

set -e
source scripts/geneval/config_dalle.sh

# Step 1: Extract GenEval prompts, grouped by tag
if [ -f "$PROMPTS_FILE" ]; then
    echo "$PROMPTS_FILE already exists. Skipping extraction."
else
    echo "Extracting prompts from $METADATA_FILE..."
    python scripts/extract/extract_prompts.py
fi

# Step 2: Run the TIR refinement loop on the GenEval prompts
echo "Running the DALL-E pipeline on GenEval prompts..."
python src/run_dalle.py \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --size "$SIZE" \
    --quality "$QUALITY" \
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