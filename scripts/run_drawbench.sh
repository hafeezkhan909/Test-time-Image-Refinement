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
python src/main.py --model_version 2.1 --prompts_file drawbench_prompts.json --output_dir drawbench/drawbench_raw_images

# Step 4: Post-process the Generated Images
echo "🔹 Step 4: Post-processing generated images for DrawBench evaluation..."
python scripts/post_process/post_process_drawbench.py --output_dir drawbench/drawbench_gen_images --generated_dir drawbench/drawbench_raw_images

# Step 5: Evaluate the Generated Images
echo "🔹 Step 5: Evaluating generated images using DrawBench metrics..."
python drawbench/drawbench_eval.py --image_folder drawbench/drawbench_gen_images --prompt_json image_prompt_mapping.json

# Final Status
echo "✅ Pipeline completed successfully!"
