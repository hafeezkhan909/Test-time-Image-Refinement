import os
import json
import argparse
import requests
from io import BytesIO
import base64
from PIL import Image
import time
import shutil
import random
import math
# Choose one of these imports based on your preference
from qwen_integration import get_refined_prompt
# from aoai import get_refined_prompt

# Initialize OpenAI client
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
dalle_model = os.getenv("DALLE_MODEL", "dall-e-3")

# ======================== #
# 🔹 Argument Parsing
# ======================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Run the DALL-E pipeline with prompt refinement.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="filtered_prompts.json",
        help="Path to the prompts file (default: filtered_prompts.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dalle_outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--refinement_iterations",
        type=int,
        default=3,
        help="Number of refinement iterations to perform (default: 3)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        choices=["1024x1024", "1792x1024", "1024x1792"],
        help="Image size (default: 1024x1024)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="standard",
        choices=["standard", "hd"],
        help="Image quality (default: standard)"
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
    """Parses Qwen/AOAI output to extract decision (True/False) and refined prompt."""
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

def check_image_exists(filepath):
    """Check if an image already exists at the given path."""
    return os.path.exists(filepath)

def generate_image_dalle(prompt, output_path, size="1024x1024", quality="standard", style="vivid"):
    """
    Generate an image using DALL-E and save it to the specified path.
    
    Args:
        prompt (str): The prompt to generate an image from.
        output_path (str): Path to save the generated image.
        size (str): Size of the image to generate.
        quality (str): Quality of the image ("standard" or "hd").
        style (str): Style of the image ("vivid" or "natural").
        
    Returns:
        bool: True if image was successfully generated, False otherwise.
    """
    if check_image_exists(output_path):
        print(f"✅ Image already exists at {output_path}. Skipping generation.")
        return True
        
    try:
        # Generate image using DALL-E
        response = client.images.generate(
            model=dalle_model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )
        
        # Extract image URL from response
        image_url = response.data[0].url
        
        # Download the image
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        
        print(f"✅ Image saved to {output_path}")
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        return True
        
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        return False

# ======================== #
# 🔹 Tag-Specific Logic
# ======================== #
def process_tag(tag, prompts, output_dir, refinement_iterations, size, quality):
    """Process all prompts for a specific tag."""
    print(f"\n🔹 Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]
        print(f"\n🚀 Processing Prompt {line_number} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder using the line number
        prompt_id = f"prompt_{line_number:03d}"
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        # ----------------------- #
        # 🔹 Step 1: Initial Image Generation
        # ----------------------- #
        initial_image_path = os.path.join(prompt_output_dir, "initial_image.png")
        generate_image_dalle(prompt, initial_image_path, size, quality)
        
        # ========================== #
        # 🔹 Refinement Loop
        # ========================== #
        current_prompt = prompt
        prompt_history = []  # Track prompt history
        
        for i in range(refinement_iterations):
            # Get previous image path
            if i == 0:
                prev_image_path = initial_image_path
            else:
                prev_image_path = os.path.join(prompt_output_dir, f"refined_image_{i}.png")
            
            # Check if refined prompt already exists and load it if it does
            refined_prompt_path = os.path.join(prompt_output_dir, f"refined_prompt_{i+1}.txt")
            refined_image_path = os.path.join(prompt_output_dir, f"refined_image_{i+1}.png")
            
            if check_image_exists(refined_prompt_path) and check_image_exists(refined_image_path):
                print(f"✅ Refined prompt and image {i+1} already exist. Loading from file.")
                with open(refined_prompt_path, "r") as f:
                    full_output = f.read()
                decision, refined_prompt = parse_qwen_output(full_output)
            else:
                # Pass the original prompt and history to get_refined_prompt
                full_output = get_refined_prompt(prompt, prev_image_path, tag, prompt_history)
                decision, refined_prompt = parse_qwen_output(full_output)
                # Save refined prompt
                with open(refined_prompt_path, "w") as f:
                    f.write(full_output)
            
            print(f"✅ Refinement {i+1}: {refined_prompt}")
            
            # Add current refinement to history
            prompt_history.append(refined_prompt)
            
            # Check if early stopping is triggered
            if decision == "True":
                print(f"✅ Early stopping at iteration {i+1}, image matches the prompt.")
                es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                if not check_image_exists(es_final_path):
                    shutil.copy(prev_image_path, es_final_path)
                break
            
            # Generate new image with refined prompt
            if not check_image_exists(refined_image_path):
                generate_image_dalle(refined_prompt, refined_image_path, size, quality)
            
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
        process_tag(tag, prompts, args.output_dir, args.refinement_iterations, args.size, args.quality)

    print("\n✅ Batch Processing Complete for all tags!")

if __name__ == "__main__":
    main()
