import os
import json
import argparse
import httpx
import time
import shutil

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://dalle-exploration.openai.azure.com/")  

# Initialize Azure OpenAI Service client with Entra ID authentication
token_provider = get_bearer_token_provider(  
    DefaultAzureCredential(),  
    "https://cognitiveservices.azure.com/.default"  
)  
  
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    azure_ad_token_provider=token_provider,  
    api_version="2024-05-01-preview",  
)


dalle_model = os.environ.get("DALLE_MODEL", "dalle3")

# ======================== #
#    Argument Parsing
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
    parser.add_argument(
        "--mllm",
        type=str,
        default="qwen",
        choices=["qwen", "gpt4o"],
        help="Which model judges/refines the prompt each round (default: qwen)"
    )
    return parser.parse_args()

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
        print(f"Image already exists at {output_path}. Skipping generation.")
        return True
        
    try:
        # Generate image using DALL-E
        result = client.images.generate(
            model=dalle_model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )
        
        # Extract image URL from response
        json_response = json.loads(result.model_dump_json())
        image_url = json_response["data"][0]["url"]
        
        # Download the image
        generated_image = httpx.get(image_url).content
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as image_file:
            image_file.write(generated_image)
        
        print(f"Image saved to {output_path}")
        # Add a small delay to avoid rate limiting
        time.sleep(1)
        return True
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return False

# ======================== #
#    Main Processing Loop
# ======================== #
def process_tag(tag, prompts, output_dir, refinement_iterations, size, quality, get_refined_prompt):
    """Process all prompts for a specific tag."""
    print(f"\nProcessing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]
        print(f"\nProcessing Prompt {line_number} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder using the line number
        prompt_id = f"prompt_{line_number:03d}"
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        # ----------------------- #
        #    Step 1: Initial Image Generation
        # ----------------------- #
        final_image_path = os.path.join(prompt_output_dir, "final_image.png")
        if not generate_image_dalle(prompt, final_image_path, size, quality):
            print(f"Skipping prompt {line_number} for tag {tag}: initial image generation failed.")
            continue
        
        # ========================== #
        #    Refinement Loop
        # ========================== #
        prompt_history = []  # Track prompt history
        
        for i in range(refinement_iterations):
            # Get previous image path
            if i == 0:
                prev_image_path = final_image_path
            else:
                prev_image_path = os.path.join(prompt_output_dir, f"final_{i}_refined_0_.png")
            
            # Check if refined prompt already exists and load it if it does
            refined_prompt_path = os.path.join(prompt_output_dir, f"{i}_refined_prompt_0.txt")
            refined_image_path = os.path.join(prompt_output_dir, f"final_{i+1}_refined_0_.png")
            
            if check_image_exists(refined_prompt_path) and check_image_exists(refined_image_path):
                print(f"Refined prompt and image {i+1} already exist. Loading from file.")
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
            
            print(f"Refinement {i+1}: {refined_prompt}")
            
            # Add current refinement to history
            prompt_history.append(refined_prompt)
            
            # Check if the image is already faithful to the prompt
            if decision == "True":
                print(f"Faithful image produced at iteration {i+1}, marked as ES{i+1} (refinement continues).")
                es_final_path = os.path.join(prompt_output_dir, f"ES{i+1}_final_image.png")
                if not check_image_exists(es_final_path):
                    shutil.copy(prev_image_path, es_final_path)
                # Removed the break statement to continue generating images
            
            # Generate new image with refined prompt
            if not check_image_exists(refined_image_path):
                if not generate_image_dalle(refined_prompt, refined_image_path, size, quality):
                    print(f"Stopping refinement for prompt {line_number} for tag {tag} at iteration {i+1}: image generation failed.")
                    break
            
    print(f"\nCompleted processing for tag: {tag}")

# ======================== #
#    Main Pipeline
# ======================== #
def main():
    args = parse_args()
    get_refined_prompt = load_refiner(args.mllm)

    # Load prompts grouped by tag
    grouped_prompts = load_prompts(args.prompts_file)

    # Process each tag separately
    for tag, prompts in grouped_prompts.items():
        process_tag(tag, prompts, args.output_dir, args.refinement_iterations, args.size, args.quality, get_refined_prompt)

    print("\nBatch Processing Complete for all tags!")

if __name__ == "__main__":
    main()