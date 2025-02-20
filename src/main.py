# src/main.py
import os
import json
import torch
import argparse
from diffusion.pipeline import generate_image
from qwen_integration import get_refined_prompt
from diffusion.refine import refine_image
import random

# ======================== #
# 🔹 Argument Parsing
# ======================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Run the diffusion pipeline with a specified model version.")
    parser.add_argument(
        "--model_version",
        type=str,
        default="1.4",
        choices=["1.4", "1.5", "2.1"],
        help="Stable Diffusion model version to use (default: 1.4)"
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
        default="new_outputs/batch_results_test",
        help="Directory to save outputs (default: new_outputs/batch_results_test)"
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
def process_tag(tag, prompts, output_dir, model_version):
    """Process all prompts for a specific tag."""
    print(f"\n🔹 Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        print(f"\n🚀 Processing Prompt {idx+1}/{len(prompts)} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder
        prompt_id = f"prompt_{idx+1:03d}"
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        # ----------------------- #
        # 🔹 Step 1: Initial Image Generation (Step 0 → Step 100)
        # ----------------------- #
        generator_seed_1 = random.randint(0, 1000000)
        generator_seed_1, latents_dict = generate_image(
            prompt, 
            generator_seed=generator_seed_1, 
            save_intermediate_steps=True,
            output_dir=prompt_output_dir,
            model_version=model_version
        )
        
        # Append the generator seed to the prompt-specific seed file
        seed_file = os.path.join(prompt_output_dir, "seed.txt")
        with open(seed_file, "a") as f:
            f.write(f"{generator_seed_1}\n")

        # Ensure required files exist
        if 10 not in latents_dict or not os.path.exists(os.path.join(prompt_output_dir, "step_75.png")):
            print(f"❌ Skipping {prompt_id}, missing required latent/image files.")
            continue

        # ----------------------- #
        # 🔹 Step 2: First Refinement (Step 25 → Step 100)
        # ----------------------- #
        full_output_25 = get_refined_prompt(prompt, os.path.join(prompt_output_dir, "step_75.png"), tag)
        decision_25, refined_prompt_25 = parse_qwen_output(full_output_25)

        # Save Qwen output
        refined_prompt_25_file = os.path.join(prompt_output_dir, "refined_prompt_25.txt")
        with open(refined_prompt_25_file, "w") as f:
            f.write(full_output_25)

        if decision_25 == "True":
            print(f"✅ Early stopping at 1st iter: Image matches the prompt.")
            move_files({os.path.join(prompt_output_dir, "final_image.png"): os.path.join(prompt_output_dir, "ES1_final_image.png")})
            # continue

        if refined_prompt_25 == "None":
            with open(os.path.join(prompt_output_dir, "dummy_25.txt"), "w") as f:
                f.write("ES1_final_image.png")  # Write the content inside the file

        print(f"✅ Refining further: Refined Prompt for Step 25: {refined_prompt_25}")

        if 25 in latents_dict:
            _, saved_latents_25 = refine_image(
                refined_prompt_25, 
                latents_dict[25], 
                start_timestep=25, 
                save_intermediate_steps=True, 
                save_steps={38, 49, 57},
                output_dir=prompt_output_dir,
                prefix="refined_25_",
                model_version=model_version
            )

        # ----------------------- #
        # 🔹 Step 3: Second Refinement (Step 10 → Step 100)
        # ----------------------- #
        refined_25_step_75_image = os.path.join(prompt_output_dir, "refined_25_step_75.png")
        full_output_10 = get_refined_prompt(prompt, refined_25_step_75_image, tag)
        decision_10, refined_prompt_10 = parse_qwen_output(full_output_10)

        refined_prompt_10_file = os.path.join(prompt_output_dir, "refined_prompt_10.txt")
        with open(refined_prompt_10_file, "w") as f:
            f.write(full_output_10)

        if decision_10 == "True":
            print(f"✅ Early stopping at 2nd iter: Image matches the prompt.")
            move_files({os.path.join(prompt_output_dir, "final_refined_25_.png"): os.path.join(prompt_output_dir, "ES2_final_image.png")})
            # continue

        if refined_prompt_10 == "None":
            with open(os.path.join(prompt_output_dir, "dummy_10.txt"), "w") as f:
                f.write("ES2_final_image.png")  # Write the content inside the file

        print(f"✅ Refining further: Refined Prompt for Step 10: {refined_prompt_10}")
        
        if 10 in latents_dict:
            _, saved_latents_10 = refine_image(
                refined_prompt_10, 
                latents_dict[10], 
                start_timestep=10, 
                save_intermediate_steps=True, 
                save_steps={14, 23, 45, 59, 68},
                output_dir=prompt_output_dir,
                prefix="refined_10_",
                model_version=model_version
            )

        # ----------------------- #
        # 🔹 Step 4: Final Generation with Latest Refined Prompt
        # ----------------------- #
        refined_10_step_75_image = os.path.join(prompt_output_dir, "refined_10_step_75.png")
        full_output_final = get_refined_prompt(prompt, refined_10_step_75_image, tag)
        decision_final, refined_prompt_final = parse_qwen_output(full_output_final)

        refined_prompt_final_file = os.path.join(prompt_output_dir, "refined_prompt_final.txt")
        with open(refined_prompt_final_file, "w") as f:
            f.write(full_output_final)

        if decision_final == "True":
            print(f"✅ Final image is satisfactory. No further refinement needed.")
            move_files({os.path.join(prompt_output_dir, "final_refined_10_.png"): os.path.join(prompt_output_dir, "ES3_final_image.png")})
            # continue

        if refined_prompt_final == "None":
            with open(os.path.join(prompt_output_dir, "dummy_final.txt"), "w") as f:
                f.write("ES3_final_image.png")  # Write the content inside the file

        print(f"✅ Generating final image with refined prompt: {refined_prompt_final}")
        generator_seed_2 = random.randint(0, 1000000)
        generate_image(
            refined_prompt_final, 
            generator_seed=generator_seed_2, 
            save_intermediate_steps=True,
            output_dir=prompt_output_dir,
            prefix="final_",
            model_version=model_version
        )

        # Append the second generator seed to the same seed file
        with open(seed_file, "a") as f:
            f.write(f"{generator_seed_2}\n")

        move_files({"outputs/final/final_image.png": os.path.join(prompt_output_dir, "final_final_image.png")})
        print(f"🎉 Completed {prompt_id} for tag {tag}! Results saved in {prompt_output_dir}")

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
        process_tag(tag, prompts, args.output_dir, args.model_version)

    print("\n✅ Batch Processing Complete for all tags!")

if __name__ == "__main__":
    main()