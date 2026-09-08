#!/bin/bash
# Runs run_flux.py against the custom T2I benchmark (negation, counting,
# position, color_attr), then scores each category.
#
# FLUX.1-dev is a gated checkpoint on Hugging Face. Accept its license and
# authenticate locally (huggingface-cli login or HF_TOKEN) before running.

set -e
source scripts/t2i_benchmark/config_flux.sh

# Step 1: Build the query JSONs the evaluation scripts read (skip if already built)
if [ -f "$QUERY_JSON_DIR/negation_specific.json" ]; then
    echo "Query JSONs already exist in $QUERY_JSON_DIR. Skipping."
else
    echo "Building query JSONs from $DATASET_FILE..."
    python scripts/extract/build_query_jsons.py --dataset "$DATASET_FILE" --output_dir "$QUERY_JSON_DIR"
fi

# Step 2: Run the TIR refinement loop on the benchmark prompts
echo "Running the Flux pipeline on the T2I benchmark..."
python src/run_flux.py \
    --prompts_file "$DATASET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --mllm "$MLLM"

# Step 3: Score each category
echo "Scoring negation..."
python evaluation/test_negation.py \
    --query_json "$QUERY_JSON_DIR/negation_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/negation_results.json" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo "Scoring counting..."
python evaluation/test_generative_numeracy.py \
    --query_json "$QUERY_JSON_DIR/counting_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/counting_results.json" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo "Scoring position..."
python evaluation/test_spatial_relationships.py \
    --query_json "$QUERY_JSON_DIR/position_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/position_results.json" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo "Scoring color_attr..."
python evaluation/test_object_detection.py \
    --query_json "$QUERY_JSON_DIR/color_attr_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/color_attr_results.json" \
    --refinement_iterations "$REFINEMENT_ITERATIONS" \
    --last_restart_step 0 \
    --selection_mode "$SELECTION_MODE"

echo ""
echo "Done. Results are in $RESULTS_DIR."