
# ===========================================================
# Full Pipeline to Generate & Evaluate DrawBench Images with DALL-E
# ===========================================================

# Step 1: Extract Prompts from DrawBench
echo "🔹 Step 1: Extracting prompts from DrawBench dataset..."
# python src/prepare_drawbench_prompts.py  # Creates drawbench_prompts.json

# Step 2: Generate Image-Prompt Mapping
echo "🔹 Step 2: Mapping prompts to expected image names..."
python src/image_prompt_mapping.py  # Creates image_prompt_mapping.json

# Step 3: Run the DALL-E Pipeline (Image Generation)
echo "🔹 Step 3: Running the DALL-E pipeline for image generation..."
OUTPUT_DIR="drawbench/drawbench_dalle_raw_images"
DRAWBENCH_OUTPUT_DIR="drawbench/drawbench_dalle_gen_images"
REFINEMENT_ITERATIONS=2

# # Make sure environment variables for Azure OpenAI are set
# if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
#   echo "⚠️ Warning: AZURE_OPENAI_ENDPOINT environment variable is not set"
#   echo "Please set it using: export AZURE_OPENAI_ENDPOINT=your_endpoint_url"
#   exit 1
# fi

# Run the DALL-E generation pipeline
python src/modular/main_dalle.py \
  --prompts_file drawbench_prompts.json \
  --output_dir "$OUTPUT_DIR" \
  --refinement_iterations $REFINEMENT_ITERATIONS \
  --size "1024x1024" \
  --quality "standard"

exit 0
# Step 4: Post-process the Generated Images
echo "🔹 Step 4: Post-processing generated images for DrawBench evaluation..."
FINAL_IMAGE_NAME="refined_image_${REFINEMENT_ITERATIONS}.png"  # Use the last refined image
python scripts/post_process/post_process_drawbench.py \
  --final_image "$FINAL_IMAGE_NAME" \
  --output_dir "$DRAWBENCH_OUTPUT_DIR" \
  --generated_dir "$OUTPUT_DIR"

# Step 5: Evaluate the Generated Images
echo "🔹 Step 5: Evaluating generated images using DrawBench metrics..."
python drawbench/drawbench_eval.py \
  --image_folder "$DRAWBENCH_OUTPUT_DIR" \
  --prompt_json image_prompt_mapping.json

# Final Status
echo "✅ DALL-E DrawBench pipeline completed successfully!"