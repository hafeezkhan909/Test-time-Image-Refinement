#!/bin/bash
# Runs the TIR pipeline against the custom T2I benchmark (negation, counting,
# position, color_attr), then scores each category with the matching evaluation script.

set -e
source scripts/t2i_benchmark/config_stable_diffusion_1_2.sh

LAST_RESTART_STEP="${RESTART_STEPS[${#RESTART_STEPS[@]}-1]}"

# Step 1: Build the query JSONs the evaluation scripts read (skip if already built)
if [ -f "$QUERY_JSON_DIR/negation_specific.json" ]; then
    echo "Query JSONs already exist in $QUERY_JSON_DIR. Skipping."
else
    echo "Building query JSONs from $DATASET_FILE..."
    python scripts/extract/build_query_jsons.py --dataset "$DATASET_FILE" --output_dir "$QUERY_JSON_DIR"
fi

# Step 2: Run the TIR refinement loop on the benchmark prompts
echo "Running the diffusion pipeline (Stable Diffusion $MODEL_VERSION) on the T2I benchmark..."
python src/run_stable_diffusion_1_2.py \
    --model_version "$MODEL_VERSION" \
    --prompts_file "$DATASET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --restart_steps "${RESTART_STEPS[@]}" \
    --refinement_step "$REFINEMENT_STEP" \
    --mllm "$MLLM"

# Step 3: Score each category
echo "Scoring negation..."
python evaluation/test_negation.py \
    --query_json "$QUERY_JSON_DIR/negation_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/negation_results.json" \
    --refinement_iterations "${#RESTART_STEPS[@]}" \
    --last_restart_step "$LAST_RESTART_STEP" \
    --selection_mode "$SELECTION_MODE"

echo "Scoring counting..."
python evaluation/test_generative_numeracy.py \
    --query_json "$QUERY_JSON_DIR/counting_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/counting_results.json" \
    --refinement_iterations "${#RESTART_STEPS[@]}" \
    --last_restart_step "$LAST_RESTART_STEP" \
    --selection_mode "$SELECTION_MODE"

echo "Scoring position..."
python evaluation/test_spatial_relationships.py \
    --query_json "$QUERY_JSON_DIR/position_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/position_results.json" \
    --refinement_iterations "${#RESTART_STEPS[@]}" \
    --last_restart_step "$LAST_RESTART_STEP" \
    --selection_mode "$SELECTION_MODE"

echo "Scoring color_attr..."
python evaluation/test_object_detection.py \
    --query_json "$QUERY_JSON_DIR/color_attr_specific.json" \
    --images_dir "$OUTPUT_DIR" \
    --output_json "$RESULTS_DIR/color_attr_results.json" \
    --refinement_iterations "${#RESTART_STEPS[@]}" \
    --last_restart_step "$LAST_RESTART_STEP" \
    --selection_mode "$SELECTION_MODE"

echo ""
echo "Done. Results are in $RESULTS_DIR."