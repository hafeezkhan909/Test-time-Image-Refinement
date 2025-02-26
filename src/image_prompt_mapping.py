import json

def create_image_prompt_mapping(input_path="drawbench_prompts.json", output_path="image_prompt_mapping.json"):
    """Creates a JSON file mapping image names (prompt_XXX) to their respective prompts."""
    
    print("\n🔹 Loading DrawBench prompts...")
    with open(input_path, "r") as f:
        categorized_prompts = json.load(f)

    image_prompt_mapping = {}

    print("\n✅ Generating Image-to-Prompt Mapping...")

    # Iterate through categories and create mapping
    for category, prompts in categorized_prompts.items():
        for prompt_entry in prompts:
            line_number = prompt_entry["line_number"]
            prompt_text = prompt_entry["prompt"]
            
            # Format the image name as "prompt_XXX"
            image_name = f"prompt_{line_number:03d}"  # Change extension as needed
            
            image_prompt_mapping[image_name] = prompt_text

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(image_prompt_mapping, f, indent=4)

    print(f"\n✅ Image-to-Prompt Mapping saved successfully in `{output_path}`!")

if __name__ == "__main__":
    create_image_prompt_mapping()
