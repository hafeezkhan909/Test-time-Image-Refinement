#!/bin/bash
# Runs run_flux.py against the GenEval benchmark prompts,
# then reorganizes the output into the folder structure GenEval expects.
#
# FLUX.1-dev is a gated checkpoint on Hugging Face. Accept its license and
# authenticate locally (huggingface-cli login or HF_TOKEN) before running.
#
# Note: GenEval itself is not bundled in this repo. Clone it separately
# (https://github.com/djghosh13/geneval) and run its own evaluation/evaluate_images.py
# against $GENEVAL_OUTPUT_DIR once this script is done executing.

set -e
source scripts/geneval/config_flux.sh

# ----------------------- #
#    Step 1: Extract GenEval prompts, grouped by tag
# ----------------------- #
if [ -f "$PROMPTS_FILE" ]; then
    echo "$PROMPTS_FILE already exists. Skipping extraction."
else
    echo "Extracting prompts from $METADATA_FILE..."
    python scripts/extract/extract_prompts.py
fi

# ----------------------- #
#    Step 2: Run the TIR refinement loop on the GenEval prompts
# ----------------------- #
echo "Running the Flux pipeline on GenEval prompts..."
python src/run_flux.py \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --mllm "$MLLM"

# ----------------------- #
#    Step 3: Reorganize output into GenEval's expected structure
# ----------------------- #
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