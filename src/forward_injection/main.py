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
        default=[25, 50, 75],
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
            restart_step=restart_steps[0],
            refinement_step=refinement_step
        )

        # ========================== #
        # 🔹 Refinement Loop
        # ========================== #
        current_prompt = prompt
        next_restart_latent = latents_dict  # Stores the latent at the next restart step
        next_offset_latent_step = None  # This will store the mapped latent step
        offset_latent_step = None
        for i, restart_step in enumerate(restart_steps):

            # Compute adjusted refinement step
            remaining_steps = 100 - restart_step
            adjusted_refinement_step = math.ceil(refinement_step * (remaining_steps / 100))

            if i == 0: 
                prev_step_image = os.path.join(prompt_output_dir, f"step_{refinement_step}.png")
            else:
                prev_restart_step = restart_steps[i - 1]  # Get the previous restart step
                prev_step_image = os.path.join(prompt_output_dir, f"final_{i}_refined_{prev_restart_step}_.png")

            full_output = get_refined_prompt(current_prompt, prev_step_image, tag)
            decision, refined_prompt = parse_qwen_output(full_output)
            print(f"✅ Refining further: Refined Prompt for Step {restart_step}: {refined_prompt}")
            
            # Save refined prompt
            with open(os.path.join(prompt_output_dir, f"{i}_refined_prompt_{restart_step}.txt"), "w") as f:
                f.write(full_output)

            if i == 0:
              if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    move_files({os.path.join(prompt_output_dir, "final_image.png"): os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")})
            else:
                if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    move_files({os.path.join(prompt_output_dir, f"final_refined_{prev_restart_step}_.png"): os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")})

            current_prompt = refined_prompt  # Update refined prompt for the next iteration

            # Compute offset latent step for the next restart step (mapped to the refinement run)
            if i < len(restart_steps) - 1:
                next_restart_step = restart_steps[i + 1]  # Next restart step
                current_remaining_steps = 100 - restart_step  # Remaining steps in current refinement
                offset_latent_step = math.ceil((next_restart_step * current_remaining_steps) / 100)
            else:
                offset_latent_step = None  # No more restart steps left

            # Refinement run using the current step's latents
            _, new_latents = refine_image(
                current_prompt,
                next_restart_latent[next_offset_latent_step] if i > 0 else next_restart_latent[restart_step],
                start_timestep=restart_step,  
                save_intermediate_steps=True,
                refinement_step=refinement_step,
                adjusted_refinement_step=adjusted_refinement_step,
                output_dir=prompt_output_dir,
                prefix=f"{i+1}_refined_{restart_step}_",
                model_version=model_version,
                n_offset_latent_step=offset_latent_step
            )
            
            # Update stored latents and mapped step for the next restart iteration
            next_restart_latent = new_latents
            next_offset_latent_step = offset_latent_step
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