#!/bin/bash
# Run the main pipeline

# Activate virtual environment
source venv/bin/activate

# Step 1: Extract prompts from evaluation_metadata.jsonl
echo "🔹 Extracting prompts from evaluation_metadata.jsonl..."
python scripts/extract/extract_prompts.py

# Step 2: Run the main script with default configuration
echo "🔹 Running the diffusion pipeline..."
python src/main.py

# Step 3: Post-process the generated images
echo "🔹 Post-processing generated images. Getting the images ready for GenEval test..."
python scripts/post_process/post_process.py