import os
import json
import shutil
import argparse

def post_process_drawbench(output_dir, generated_dir):
    """
    Reorganizes generated images from multiple categories into a single folder.

    Args:
        output_dir (str): Path to the output directory where images will be saved.
        generated_dir (str): Path to the directory containing the generated images, structured by category.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prioritized image selection order
    image_priority = [
        "ES1_final_image.png",  # Select the image from early stopping after 1st iter
        "ES2_final_image.png",  # Select the image from early stopping after 2nd iter
        "ES3_final_image.png",  # Select the image from early stopping after 3rd iter
        "final_final_image.png"  # Reached final step
    ]

    print("\n🔹 Processing generated images...")

    # Iterate over categories (subfolders in generated_dir)
    for category in os.listdir(generated_dir):
        category_path = os.path.join(generated_dir, category)

        if not os.path.isdir(category_path):
            continue  # Skip non-folder files

        # Iterate over prompt folders (e.g., prompt_001, prompt_002, etc.)
        for prompt_folder in os.listdir(category_path):
            prompt_folder_path = os.path.join(category_path, prompt_folder)

            if not os.path.isdir(prompt_folder_path):
                continue  # Skip non-folder files

            # Extract prompt number (prompt_XXX)
            prompt_id = prompt_folder  # e.g., "prompt_002"

            # Find the highest priority available image
            selected_image = None
            for image_name in image_priority:
                image_path = os.path.join(prompt_folder_path, image_name)
                if os.path.exists(image_path):
                    selected_image = image_path
                    break

            # Copy the selected image to the output directory
            if selected_image:
                new_image_name = f"{prompt_id}.png"  # Save as prompt_XXX.png
                new_image_path = os.path.join(output_dir, new_image_name)
                shutil.copy(selected_image, new_image_path)
                print(f"✅ Saved {new_image_name}")
            else:
                print(f"❌ No final image found for {prompt_id}")

    print(f"\n✅ Post-processing complete. All images saved in `{output_dir}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process generated images for DrawBench.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save the processed images.")
    parser.add_argument("--generated_dir", required=True, type=str, help="Path to the folder containing generated images.")

    args = parser.parse_args()

    post_process_drawbench(args.output_dir, args.generated_dir)
