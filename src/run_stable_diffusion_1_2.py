import os
import json
import torch
import argparse
from diffusion.pipeline import generate_image
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
        default="new_outputs/geneval_batch_results_SD",
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
        help="Step at which to take feedback from the MLLM (default: 99)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generator seed used for every image generation call (default: 42)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of denoising steps for the diffusion schedule (default: 100)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height in pixels (default: 512)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width in pixels (default: 512)"
    )
    parser.add_argument(
        "--mllm",
        type=str,
        default="qwen",
        choices=["qwen", "gpt4o"],
        help="Which model judges/refines the prompt each round: local Qwen2.5-VL or Azure OpenAI GPT-4o (default: qwen)"
    )
    return parser.parse_args()

# ======================== #
#    MLLM Selection
# ======================== #
def load_refiner(mllm):
    if mllm == "qwen":
        from qwen_integration import get_refined_prompt
    elif mllm == "gpt4o":
        from aoai import get_refined_prompt
    else:
        raise ValueError(f"Unknown mllm choice: {mllm!r}")
    return get_refined_prompt

# ======================== #
#    Utility Functions
# ======================== #
def load_prompts(file_path):
    """Loads prompts from JSON file grouped by tag."""
    with open(file_path, "r") as f:
        return json.load(f)

def parse_qwen_output(full_output):
    """Parses the MLLM's output to extract decision (True/False) and refined prompt."""
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
#    Main Processing Loop
# ======================== #
def process_tag(tag, prompts, output_dir, model_version, restart_steps, refinement_step, seed,
                 num_inference_steps, guidance_scale, height, width, get_refined_prompt):
    """Process all prompts for a specific tag."""
    print(f"\n   Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)

    for prompt_data in prompts:

        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]  # Get the line number
        print(f"\nProcessing Prompt {line_number} for tag {tag}:\n{prompt}")

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
            print(f"Initial image already exists for prompt {line_number}. Skipping generation.")
            # We need to load the latents_dict if it exists
            latents_dict = {}
            for step in restart_steps:
                latent_path = os.path.join(prompt_output_dir, f"latents_{step}.pt")
                if os.path.exists(latent_path):
                    latents_dict[step] = torch.load(latent_path)
            generator_seed_1 = seed
        else:
            generator_seed_1 = seed
            generator_seed_1, latents_dict = generate_image(
                prompt, 
                generator_seed=generator_seed_1, 
                save_intermediate_steps=True,
                output_dir=prompt_output_dir,
                model_version=model_version,
                restart_steps=restart_steps,
                prefix="image",
                refinement_step=refinement_step,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )

        # ========================== #
        #    Refinement Loop
        # ========================== #
        prompt_history = []  # Track prompt history
        
        for i, restart_step in enumerate(restart_steps):
            # Compute adjusted refinement step
            remaining_steps = num_inference_steps - restart_step
            adjusted_refinement_step = math.floor(refinement_step * (remaining_steps / num_inference_steps))

            if i == 0: 
                prev_step_image = os.path.join(prompt_output_dir, f"imagestep_{refinement_step}.png")
            else:
                prev_restart_step = restart_steps[i - 1]  # Get the previous restart step
                prev_step_image = os.path.join(prompt_output_dir, f"final_{i}_refined_{prev_restart_step}_.png")

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
                    print(f"Faithful image produced at iteration {i+1}, marked as ES{i+1} (refinement continues).")
                    es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                    if not check_image_exists(es_final_path):
                        shutil.copy(os.path.join(prompt_output_dir, "final_image.png"), es_final_path)
            else:
                if decision == "True":
                    print(f"Faithful image produced at iteration {i+1}, marked as ES{i+1} (refinement continues).")
                    es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                    prev_final_path = os.path.join(prompt_output_dir, f"final_{i}_refined_{prev_restart_step}_.png")
                    if not check_image_exists(es_final_path) and check_image_exists(prev_final_path):
                        shutil.copy(prev_final_path, es_final_path)

            # Check if refined image already exists
            refined_image_path = os.path.join(prompt_output_dir, f"final_{i+1}_refined_{restart_step}_.png")

            if check_image_exists(refined_image_path):
                print(f"Refined image for step {restart_step} already exists. Skipping generation.")
                continue

            # Restart generation (either from 0 or using refinement)
            if restart_step == 0:
                generator_seed_2 = seed
                generate_image(
                    refined_prompt, 
                    generator_seed=generator_seed_2, 
                    save_intermediate_steps=True,
                    output_dir=prompt_output_dir,
                    prefix=f"{i+1}_refined_{restart_step}_",
                    model_version=model_version,
                    refinement_step=refinement_step,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width
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
                    model_version=model_version,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
    print(f"\nCompleted processing for tag: {tag}")
    

# ======================== #
#    Main Pipeline
# ======================== #
def main():
    args = parse_args()

    # Resolve the MLLM choice once, lazily, before any per-prompt work starts
    get_refined_prompt = load_refiner(args.mllm)

    # Load prompts grouped by tag
    grouped_prompts = load_prompts(args.prompts_file)

    # Process each tag separately
    for tag, prompts in grouped_prompts.items():
        process_tag(
            tag, prompts, args.output_dir, args.model_version, args.restart_steps, args.refinement_step,
            args.seed, args.num_inference_steps, args.guidance_scale, args.height, args.width,
            get_refined_prompt
        )

    print("\nBatch Processing Complete for all tags!")

if __name__ == "__main__":
    main()