#!/bin/bash
# Run the main pipeline

# Activate virtual environment for the main pipeline
echo "🔹 Activating virtual environment for the main pipeline..."
source venv/bin/activate

# Step 1: Extract prompts from evaluation_metadata.jsonl
echo "🔹 Extracting prompts from evaluation_metadata.jsonl..."
python scripts/extract/extract_prompts.py

# Step 2: Run the main script with SD 1.5 configuration
echo "🔹 Running the diffusion pipeline..."
python src/main.py --model_version 1.5

# Step 3: Post-process the generated images
echo "🔹 Post-processing generated images. Getting the images ready for GenEval test..."
python scripts/post_process/post_process.py

# Step 4: Activate the GenEval virtual environment
echo "🔹 Activating GenEval virtual environment..."
source geneval/geneval1/bin/activate

# Step 5: Evaluate the generated images using GenEval
echo "🔹 Evaluating generated images with GenEval..."
python geneval/evaluation/evaluate_images.py "gen_images" --outfile "results_method/results.jsonl" --model-path "./geneval/models/"

# Step 6: Generate summary scores from the evaluation results
echo "🔹 Generating summary scores..."
python geneval/evaluation/summary_scores.py "results_method/results.jsonl"

echo "✅ Pipeline completed successfully!"