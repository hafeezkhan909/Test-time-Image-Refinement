import os
import json
import shutil
import argparse

from image_selection import get_final_image_path


def post_process_geneval(metadata_file, output_dir, generated_dir, refinement_iterations, last_restart_step, selection_mode):
    """
    Reorganizes generated images and metadata into GenEval's required structure.

    Args:
        metadata_file (str): Path to the evaluation_metadata.jsonl file.
        output_dir (str): Path to the output directory (e.g. gen_images).
        generated_dir (str): Path to the directory containing the generated images.
        refinement_iterations (int): Number of refinement rounds the pipeline ran (default: 3).
        last_restart_step (int): Restart step used for the final refinement round (default: 0).
        selection_mode (str): "first_early_stop" or "final_only" -- see image_selection.get_final_image_path.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_file, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            prompt_id = f"{idx + 1:05d}"  # Format as 00001, 00002, etc.
            prompt_folder = os.path.join(output_dir, prompt_id)
            samples_folder = os.path.join(prompt_folder, "samples")
            os.makedirs(samples_folder, exist_ok=True)

            tag = data.get("tag", "unknown_tag")
            prompt = data.get("prompt", "unknown_prompt")
            generated_prompt_dir = os.path.join(generated_dir, tag, f"prompt_{idx + 1:03d}")

            selected_image = get_final_image_path(
                generated_prompt_dir,
                refinement_iterations=refinement_iterations,
                last_restart_step=last_restart_step,
                selection_mode=selection_mode,
            )

            # Copy the selected image to the samples folder
            if selected_image:
                shutil.copy(selected_image, os.path.join(samples_folder, "00000.png"))
            else:
                print(f"Missing final image for prompt {prompt_id}: {prompt}")

            # Save metadata
            metadata_path = os.path.join(prompt_folder, "metadata.jsonl")
            with open(metadata_path, "w") as f_meta:
                json.dump(data, f_meta)

    print(f"Post-processing complete. Results saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize the generated images into structured directories for GenEval.")

    parser.add_argument("--metadata_file", type=str, default="evaluation_metadata.jsonl",
                         help="Path to the evaluation metadata JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Path to the output directory where organized images will be stored.")
    parser.add_argument("--generated_dir", type=str, required=True,
                         help="Path to the directory containing the generated images.")
    parser.add_argument("--refinement_iterations", type=int, default=3,
                         help="Number of refinement rounds the pipeline ran (default: 3).")
    parser.add_argument("--last_restart_step", type=int, default=0,
                         help="Restart step used for the final refinement round (default: 0).")
    parser.add_argument("--selection_mode", type=str, default="first_early_stop",
                         choices=["first_early_stop", "final_only"],
                         help="How to pick the evaluated image per prompt (default: first_early_stop).")

    args = parser.parse_args()

    post_process_geneval(args.metadata_file, args.output_dir, args.generated_dir, args.refinement_iterations,
                          args.last_restart_step, args.selection_mode)