import os
import json
import shutil

def post_process(metadata_file, output_dir, generated_dir):
    """
    Reorganizes generated images and metadata into the required structure.
    
    Args:
        metadata_file (str): Path to the evaluation_metadata.jsonl file.
        output_dir (str): Path to the output directory (generated_images).
        generated_dir (str): Path to the directory containing the generated images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prioritized image selection order
    image_priority = [
        "ES1_final_image.png", # Select the image from early stopping after 1st iter
        "ES2_final_image.png", # Select the image from early stopping after 2nd iter
        "ES3_final_image.png", # Select the image from early stopping after 3rd iter
        "final_final_image.png" # Reached final step
    ]

    # Read the metadata file line by line
    with open(metadata_file, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            prompt_id = f"{idx + 1:05d}"  # Format as 00001, 00002, etc.
            prompt_folder = os.path.join(output_dir, prompt_id)
            samples_folder = os.path.join(prompt_folder, "samples")
            os.makedirs(samples_folder, exist_ok=True)

            # Find the corresponding generated image
            tag = data["tag"]
            prompt = data["prompt"]
            generated_prompt_dir = os.path.join(generated_dir, tag, f"prompt_{idx + 1:03d}")

            # Select the highest priority image that exists
            selected_image = None
            for image_name in image_priority:
                image_path = os.path.join(generated_prompt_dir, image_name)
                if os.path.exists(image_path):
                    selected_image = image_path
                    break

            # Copy the selected image to the samples folder
            if selected_image:
                shutil.copy(selected_image, os.path.join(samples_folder, "00000.png"))
            else:
                print(f"❌ Missing final image for prompt {prompt_id}: {prompt}")

            # Save metadata
            metadata_path = os.path.join(prompt_folder, "metadata.jsonl")
            with open(metadata_path, "w") as f_meta:
                json.dump(data, f_meta)

    print(f"✅ Post-processing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    metadata_file = "evaluation_metadata.jsonl"
    output_dir = "gen_images"
    generated_dir = "new_outputs/batch_results_test"
    post_process(metadata_file, output_dir, generated_dir)