#!/bin/bash
# Run the main pipeline

# Activate virtual environment for the main pipeline
echo "🔹 Activating virtual environment for the main pipeline..."
# source venv/bin/activate  # Uncomment this if using a virtual environment

# Step 1: Extract prompts from evaluation_metadata.jsonl
echo "🔹 Extracting prompts from evaluation_metadata.jsonl..."
# python scripts/extract/extract_prompts.py  # Uncomment if needed

# Step 2: Run the main script with SD 1.5 configuration
echo "🔹 Running the diffusion pipeline..."
RESTART_STEPS=(75 50 25 0)  # Modify this array as needed
REFINEMENT_STEP=99
OUTPUT_DIR="new_outputs/automated_batch_results"
GENEVEAL_OUTPUT_DIR="generated_images"

python src/modular/main.py --model_version 1.5 --output_dir "$OUTPUT_DIR" --restart_steps "${RESTART_STEPS[@]}" --refinement_step $REFINEMENT_STEP

# Generate the final image filename dynamically
RESTART_COUNT=${#RESTART_STEPS[@]}
LAST_RESTART=${RESTART_STEPS[-1]}
FINAL_IMAGE_NAME="final_${RESTART_COUNT}_refined_${LAST_RESTART}_.png"

# Step 3: Post-process the generated images
echo "🔹 Post-processing generated images. Getting the images ready for GenEval test..."
python scripts/post_process/post_process_mod.py --final_image "$FINAL_IMAGE_NAME" --generated_dir "$OUTPUT_DIR" --output_dir "$GENEVEAL_OUTPUT_DIR"

# Step 4: Activate the GenEval virtual environment
echo "🔹 Activating GenEval virtual environment..."
source geneval/geneval1/bin/activate  # Ensure this path is correct

# Step 5: Evaluate the generated images using GenEval
echo "🔹 Evaluating generated images with GenEval..."
python geneval/evaluation/evaluate_images.py "$GENEVEAL_OUTPUT_DIR" --outfile "results_method/results.jsonl" --model-path "./geneval/models/"

# Step 6: Generate summary scores from the evaluation results
echo "🔹 Generating summary scores..."
python geneval/evaluation/summary_scores.py "results_method/results.jsonl"

echo "✅ Pipeline completed successfully!"
