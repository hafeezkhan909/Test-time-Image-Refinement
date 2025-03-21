#!/bin/bash
# Run the main pipeline

# Activate virtual environment for the main pipeline
echo "🔹 Activating virtual environment for the main pipeline..."
source venv/bin/activate

# Step 1: Extract prompts from evaluation_metadata.jsonl
echo "🔹 Extracting prompts from evaluation_metadata.jsonl..."
python scripts/extract/extract_prompts.py

# Step 2: Run the SD XL configuration
echo "🔹 Running the SDXL diffusion pipeline..."
python src/sdxl_run.py

# Step 3: Post-process the generated images
echo "🔹 Post-processing generated images. Getting the images ready for GenEval test..."
python scripts/post_process/post_process.py # make sure to go into post_process/post_process.py and change to generated_dir = "imgs/dynamic_refinements" or desired file path on Line 63

# Step 4: Activate the GenEval virtual environment
echo "🔹 Activating GenEval virtual environment..."
source geneval/geneval1/bin/activate

# Step 5: Evaluate the generated images using GenEval
echo "🔹 Evaluating generated images with GenEval..."
python geneval/evaluation/evaluate_images.py "sdxl_gen_images" --outfile "results_sdxl/results.jsonl" --model-path "./geneval/models/" # Similarly in post_process/post_process.py, update output_dir = "sdxl_gen_images" or desired file path on Line 62

# Step 6: Generate summary scores from the evaluation results
echo "🔹 Generating summary scores..."
python geneval/evaluation/summary_scores.py "results_sdxl/results.jsonl"

echo "✅ Pipeline completed successfully!"