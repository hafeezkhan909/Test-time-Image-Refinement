import os
import json
import shutil
import argparse

def post_process(metadata_file, output_dir, generated_dir, final_image_name):
    """
    Reorganizes generated images and metadata into the required structure.

    Args:
        metadata_file (str): Path to the evaluation_metadata.jsonl file.
        output_dir (str): Path to the output directory (generated_images).
        generated_dir (str): Path to the directory containing the generated images.
        final_image_name (str): User-specified final image name (e.g., "custom_final.png").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prioritized image selection order (user-defined final image included)
    image_priority = [
        "ES1_final_image.png",  # Select the image from early stopping after 1st iter
        "ES2_final_image.png",  # Select the image from early stopping after 2nd iter
        "ES3_final_image.png",  # Select the image from early stopping after 3rd iter
        final_image_name        # User-defined final image
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
            tag = data.get("tag", "unknown_tag")
            prompt = data.get("prompt", "unknown_prompt")
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
    parser = argparse.ArgumentParser(description="Reorganize the generated images into structured directories for GenEval.")
    
    parser.add_argument(
        "--metadata_file", type=str, default="evaluation_metadata.jsonl",
        help="Path to the evaluation metadata JSONL file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the output directory where organized images will be stored."
    )
    parser.add_argument(
        "--generated_dir", type=str, required=True,
        help="Path to the directory containing the generated images."
    )
    parser.add_argument(
        "--final_image_name", type=str, default="final_final_image.png",
        help="Custom name for the final selected image (default: final_final_image.png)."
    )

    args = parser.parse_args()

    post_process(args.metadata_file, args.output_dir, args.generated_dir, args.final_image_name)
