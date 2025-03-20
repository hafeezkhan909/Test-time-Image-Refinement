# ===========================================================
# Full Pipeline to Generate & Evaluate DrawBench Images
# ===========================================================

# # Step 1: Extract Prompts from DrawBench
echo "🔹 Step 1: Extracting prompts from DrawBench dataset..."
python src/prepare_drawbench_prompts.py  # Creates drawbench_prompts.json

# # Step 2: Generate Image-Prompt Mapping
echo "🔹 Step 2: Mapping prompts to expected image names..."
python src/image_prompt_mapping.py  # Creates image_prompt_mapping.json

# # Step 3: Run the Diffusion Pipeline (Image Generation)
echo "🔹 Step 3: Running the diffusion pipeline for Stable Diffusion 1.5..."
RESTART_STEPS=(75 50 25)  # Modify this array as needed
REFINEMENT_STEP=99
OUTPUT_DIR="drawbench/drawbench_raw_images"
DRAWBENCH_OUTPUT_DIR="drawbench/drawbench_gen_images"

python src/modular/main.py --model_version 2.1 --prompts_file drawbench_prompts.json --output_dir "$OUTPUT_DIR" --restart_steps "${RESTART_STEPS[@]}" --refinement_step $REFINEMENT_STEP

# Generate the final image filename dynamically
RESTART_COUNT=${#RESTART_STEPS[@]}
LAST_RESTART=${RESTART_STEPS[-1]}
FINAL_IMAGE_NAME="final_${RESTART_COUNT}_refined_${LAST_RESTART}_.png"

# Step 4: Post-process the Generated Images
echo "🔹 Step 4: Post-processing generated images for DrawBench evaluation..."
python scripts/post_process/post_process_drawbench.py --final_image "$FINAL_IMAGE_NAME" --output_dir "$DRAWBENCH_OUTPUT_DIR" --generated_dir "$OUTPUT_DIR"

# Step 5: Evaluate the Generated Images
echo "🔹 Step 5: Evaluating generated images using DrawBench metrics..."
python drawbench/drawbench_eval.py --image_folder "$DRAWBENCH_OUTPUT_DIR" --prompt_json image_prompt_mapping.json

# Final Status
echo "✅ Pipeline completed successfully!"
