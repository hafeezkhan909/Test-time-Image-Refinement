import os
import json
import torch
import argparse
from diffusion.pipeline import generate_image
from qwen_integration import get_refined_prompt
from diffusion.refine import refine_image
import random
import math

# ======================== #
# 🔹 Argument Parsing
# ======================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Run the diffusion pipeline with user-defined restart points.")
    parser.add_argument(
        "--model_version",
        type=str,
        default="1.5",
        choices=["1.4", "1.5", "2.1"],
        help="Stable Diffusion model version to use (default: 1.5)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="filtered_prompts.json",
        help="Path to the prompts file (default: filtered_prompts.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--restart_steps",
        type=int,
        nargs="+",
        default=[25, 10, 0],
        help="List of restart steps (e.g., --restart_steps 25 10 0)"
    )
    parser.add_argument(
        "--refinement_step",
        type=int,
        default=75,
        help="Step at which to take feedback from Qwen (default: 75)"
    )
    return parser.parse_args()

# ======================== #
# 🔹 Utility Functions
# ======================== #
def load_prompts(file_path):
    """Loads prompts from JSON file grouped by tag."""
    with open(file_path, "r") as f:
        return json.load(f)

def parse_qwen_output(full_output):
    """Parses Qwen output to extract decision (True/False) and refined prompt."""
    decision = None
    refined_prompt = None

    for line in full_output.split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().strip('"')
        elif line.startswith("REFINED PROMPT:"):
            refined_prompt = line.replace("REFINED PROMPT:", "").strip().strip('"')
    # Ensure refined_prompt is valid; default to original prompt if missing
    if refined_prompt is None:
        refined_prompt = "None"
    return decision, refined_prompt

def move_files(file_map):
    """Moves files from source to destination paths."""
    for src, dest in file_map.items():
        if os.path.exists(src):
            os.rename(src, dest)

# ======================== #
# 🔹 Tag-Specific Logic
# ======================== #
def process_tag(tag, prompts, output_dir, model_version, restart_steps, refinement_step):
    """Process all prompts for a specific tag."""
    print(f"\n🔹 Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]  # Get the line number
        print(f"\n🚀 Processing Prompt {line_number} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder using the line number
        prompt_id = f"prompt_{line_number:03d}"  # Use line number for folder name
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        # ----------------------- #
        # 🔹 Step 1: Initial Image Generation (Step 0 → Step 100)
        # ----------------------- #
        generator_seed_1 = 42
        generator_seed_1, latents_dict = generate_image(
            prompt, 
            generator_seed=generator_seed_1, 
            save_intermediate_steps=True,
            output_dir=prompt_output_dir,
            model_version=model_version,
            restart_steps=restart_steps,
            refinement_step=refinement_step
        )
        
        # Append the generator seed to the prompt-specific seed file
        seed_file = os.path.join(prompt_output_dir, "seed.txt")
        with open(seed_file, "a") as f:
            f.write(f"{generator_seed_1}\n")

        # Ensure required files exist
        if 10 not in latents_dict or not os.path.exists(os.path.join(prompt_output_dir, "step_75.png")):
            print(f"❌ Skipping {prompt_id}, missing required latent/image files.")
            # continue


        # ========================== #
        # 🔹 Refinement Loop
        # ========================== #
        current_prompt = prompt
        for i, restart_step in enumerate(restart_steps):

            # Compute adjusted refinement step
            remaining_steps = 100 - restart_step
            adjusted_refinement_step = math.ceil(refinement_step * (remaining_steps / 100))

            if i == 0: 
                prev_step_image = os.path.join(prompt_output_dir, f"step_{refinement_step}.png")
            else:
                prev_restart_step = restart_steps[i - 1]  # Get the previous restart step
                prev_step_image = os.path.join(prompt_output_dir, f"refined_{prev_restart_step}_step_{refinement_step}.png")

            full_output = get_refined_prompt(current_prompt, prev_step_image, tag)
            decision, refined_prompt = parse_qwen_output(full_output)

            # Save refined prompt
            with open(os.path.join(prompt_output_dir, f"refined_prompt_{restart_step}.txt"), "w") as f:
                f.write(full_output)

            if i == 0:
              if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    move_files({os.path.join(prompt_output_dir, "final_image.png"): os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")})
            else:
                if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    move_files({os.path.join(prompt_output_dir, f"final_refined_{prev_restart_step}_.png"): os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")})

            current_prompt = refined_prompt

            # Restart generation (either from 0 or using refinement)
            if restart_step == 0:

                generator_seed_2 = 42
                generate_image(
                    refined_prompt, 
                    generator_seed=generator_seed_2, 
                    save_intermediate_steps=True,
                    output_dir=prompt_output_dir,
                    prefix=f"{i+1}_",
                    model_version=model_version,
                    refinement_step=refinement_step
                )
            else:
                _, _ = refine_image(
                    current_prompt, 
                    latents_dict[restart_step], 
                    start_timestep=restart_step, 
                    save_intermediate_steps=True, 
                    refinement_step=refinement_step,
                    adjusted_refinement_step=adjusted_refinement_step,
                    output_dir=prompt_output_dir,
                    prefix=f"refined_{restart_step}_",
                    model_version=model_version
                )
    print(f"\n✅ Completed processing for tag: {tag}")
    

# ======================== #
# 🔹 Main Pipeline
# ======================== #
def main():
    args = parse_args()

    # Load prompts grouped by tag
    grouped_prompts = load_prompts(args.prompts_file)

    # Process each tag separately
    for tag, prompts in grouped_prompts.items():
        process_tag(tag, prompts, args.output_dir, args.model_version, args.restart_steps, args.refinement_step)

    print("\n✅ Batch Processing Complete for all tags!")

if __name__ == "__main__":
    main()