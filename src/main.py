import os
import json
import torch
import argparse
from diffusion.pipeline import generate_image
from qwen_integration import get_refined_prompt
from diffusion.refine import refine_image
import math
import shutil

# ======================== #
#    Argument Parsing
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
        default="test/test",
        help="Directory to save outputs (default: test/test)"
    )
    parser.add_argument(
        "--restart_steps",
        type=int,
        nargs="+",
        default=[0, 0, 0],
        help="List of restart steps (default: 0 0 0)"
    )
    parser.add_argument(
        "--refinement_step",
        type=int,
        default=99,
        help="Step at which to take feedback from Qwen (default: 99)"
    )
    return parser.parse_args()

# ======================== #
#    Utility Functions
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

def check_image_exists(filepath):
    """Check if an image already exists at the given path."""
    return os.path.exists(filepath)

# ======================== #
#    Tag-Specific Logic
# ======================== #
def process_tag(tag, prompts, output_dir, model_version, restart_steps, refinement_step):
    """Process all prompts for a specific tag."""
    print(f"\n   Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)
    cnt = 0
    for prompt_data in prompts:

        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]  # Get the line number
        print(f"\n🚀 Processing Prompt {line_number} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder using the line number
        prompt_id = f"prompt_{line_number:03d}"  # Use line number for folder name
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        # ----------------------- #
        #    Step 1: Initial Image Generation (Step 0 → Step 100)
        # ----------------------- #
        final_image_path = os.path.join(prompt_output_dir, "final_image.png")
        refinement_image_path = os.path.join(prompt_output_dir, f"imagestep_{refinement_step}.png")
        
        if check_image_exists(final_image_path) and check_image_exists(refinement_image_path):
            print(f"✅ Initial image already exists for prompt {line_number}. Skipping generation.")
            # We need to load the latents_dict if it exists
            latents_dict = {}
            for step in restart_steps:
                latent_path = os.path.join(prompt_output_dir, f"latents_{step}.pt")
                if os.path.exists(latent_path):
                    latents_dict[step] = torch.load(latent_path)
            generator_seed_1 = 42  # Default seed
        else:
            generator_seed_1 = 42
            generator_seed_1, latents_dict = generate_image(
                prompt, 
                generator_seed=generator_seed_1, 
                save_intermediate_steps=True,
                output_dir=prompt_output_dir,
                model_version=model_version,
                restart_steps=restart_steps,
                prefix="image",
                refinement_step=refinement_step
            )

        # ========================== #
        #    Refinement Loop
        # ========================== #
        prompt_history = []  # Track prompt history
        
        for i, restart_step in enumerate(restart_steps):
            # Compute adjusted refinement step
            remaining_steps = 100 - restart_step
            adjusted_refinement_step = math.floor(refinement_step * (remaining_steps / 100))

            if i == 0: 
                # prev_step_image = os.path.join(prompt_output_dir, f"final_image.png") # if using final image
                prev_step_image = os.path.join(prompt_output_dir, f"imagestep_{refinement_step}.png") # if using final image
            else:
                prev_restart_step = restart_steps[i - 1]  # Get the previous restart step
                prev_step_image = os.path.join(prompt_output_dir, f"final_{i}_refined_{prev_restart_step}_.png") # if using final image
                # prev_step_image = os.path.join(prompt_output_dir, f"{i}_refined_{prev_restart_step}_final_image.png") # Use this when running multiple 0's

            # Check if refined prompt already exists and load it if it does
            refined_prompt_path = os.path.join(prompt_output_dir, f"{i}_refined_prompt_{restart_step}.txt")
            if check_image_exists(refined_prompt_path):
                print(f"Refined prompt for step {restart_step} already exists. Loading from file.")
                with open(refined_prompt_path, "r") as f:
                    full_output = f.read()
                decision, refined_prompt = parse_qwen_output(full_output)
            else:
                # Pass the original prompt and history to get_refined_prompt
                full_output = get_refined_prompt(prompt, prev_step_image, tag, prompt_history)
                decision, refined_prompt = parse_qwen_output(full_output)
                # Save refined prompt
                with open(refined_prompt_path, "w") as f:
                    f.write(full_output)
                    
            print(f"Refining further: Refined Prompt for Step {restart_step}: {refined_prompt}")
            
            # Add current refinement to history
            prompt_history.append(refined_prompt)
            
            if i == 0:
                if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                    if not check_image_exists(es_final_path):
                        shutil.copy(os.path.join(prompt_output_dir, "final_image.png"), es_final_path)
            else:
                if decision == "True":
                    print(f"✅ Early stopping at {i+1} iter, image matches the prompt.")
                    es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                    prev_final_path = os.path.join(prompt_output_dir, f"final_{i}_refined_{prev_restart_step}_.png")
                    if not check_image_exists(es_final_path) and check_image_exists(prev_final_path):
                        shutil.copy(prev_final_path, es_final_path)

            # Check if refined image already exists
            refined_image_path = ""
            if restart_step == 0:
                refined_image_path = os.path.join(prompt_output_dir, f"{i+1}_refined_{restart_step}_final_image.png")
            else:
                refined_image_path = os.path.join(prompt_output_dir, f"final_{i+1}_refined_{restart_step}_.png")
                
            if check_image_exists(refined_image_path):
                print(f"✅ Refined image for step {restart_step} already exists. Skipping generation.")
                continue

            # Restart generation (either from 0 or using refinement)
            if restart_step == 0:
                generator_seed_2 = 42
                generate_image(
                    refined_prompt, 
                    generator_seed=generator_seed_2, 
                    save_intermediate_steps=True,
                    output_dir=prompt_output_dir,
                    prefix=f"{i+1}_refined_{restart_step}_",
                    model_version=model_version,
                    refinement_step=refinement_step
                )
            else:
                _, _ = refine_image(
                    refined_prompt=refined_prompt, 
                    init_latents=latents_dict[restart_step], 
                    start_timestep=restart_step, 
                    save_intermediate_steps=True, 
                    refinement_step=refinement_step,
                    adjusted_refinement_step=adjusted_refinement_step,
                    output_dir=prompt_output_dir,
                    prefix=f"{i+1}_refined_{restart_step}_",
                    model_version=model_version
                )
    print(f"\n✅ Completed processing for tag: {tag}")
    

# ======================== #
#    Main Pipeline
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