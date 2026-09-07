import os
import json
import shutil
import argparse

from image_selection import get_final_image_path


def post_process_drawbench(output_dir, generated_dir, refinement_iterations, last_restart_step, selection_mode):
    """
    Reorganizes generated images from multiple categories into a single folder.

    Args:
        output_dir (str): Path to the output directory where images will be saved.
        generated_dir (str): Path to the directory containing the generated images, structured by category.
        refinement_iterations (int): Number of refinement rounds the pipeline ran (default: 3).
        last_restart_step (int): Restart step used for the final refinement round (default: 0).
        selection_mode (str): "first_early_stop" or "final_only" -- see image_selection.get_final_image_path.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nProcessing generated images...")

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

            prompt_id = prompt_folder  # e.g., "prompt_002"

            selected_image = get_final_image_path(
                prompt_folder_path,
                refinement_iterations=refinement_iterations,
                last_restart_step=last_restart_step,
                selection_mode=selection_mode,
            )

            # Copy the selected image to the output directory
            if selected_image:
                new_image_name = f"{prompt_id}.png"  # Save as prompt_XXX.png
                new_image_path = os.path.join(output_dir, new_image_name)
                shutil.copy(selected_image, new_image_path)
                print(f"Saved {new_image_name}")
            else:
                print(f"No final image found for {prompt_id}")

    print(f"\nPost-processing complete. All images saved in `{output_dir}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process generated images for DrawBench.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save the processed images.")
    parser.add_argument("--generated_dir", required=True, type=str, help="Path to the folder containing generated images.")
    parser.add_argument("--refinement_iterations", type=int, default=3,
                         help="Number of refinement rounds the pipeline ran (default: 3).")
    parser.add_argument("--last_restart_step", type=int, default=0,
                         help="Restart step used for the final refinement round (default: 0).")
    parser.add_argument("--selection_mode", type=str, default="first_early_stop",
                         choices=["first_early_stop", "final_only"],
                         help="How to pick the evaluated image per prompt (default: first_early_stop).")
    args = parser.parse_args()

    post_process_drawbench(args.output_dir, args.generated_dir, args.refinement_iterations,
                            args.last_restart_step, args.selection_mode)