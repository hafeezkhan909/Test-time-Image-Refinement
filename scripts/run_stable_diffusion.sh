#!/bin/bash
# Run the main pipeline

# Create and activate virtual environment for the main pipeline
echo "Creating and activating virtual environment for the main pipeline..."
python -m venv diff
source diff/bin/activate 

# Installing the requirements from the requirements.txt file
echo "Installing the requirements for the enviornment..."
pip install -r requirements.txt

# Step 1: Extract prompts from evaluation_metadata.jsonl
echo "Extracting prompts from evaluation_metadata.jsonl..."
python scripts/extract/extract_prompts.py

# Step 2: Run the main script with SD 1.5 configuration
echo "Running the diffusion pipeline..."
RESTART_STEPS=(0 0 0)  # Modify this array as needed
REFINEMENT_STEP=99
OUTPUT_DIR="test/test"
GENEVEAL_OUTPUT_DIR="generated_images"

python src/main.py --model_version 1.5 --output_dir "$OUTPUT_DIR" --restart_steps "${RESTART_STEPS[@]}" --refinement_step $REFINEMENT_STEP
