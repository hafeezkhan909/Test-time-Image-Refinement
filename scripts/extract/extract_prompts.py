# extract_prompts.py
import json
from collections import defaultdict

def extract_prompts(input_file, output_file):
    """
    Extracts prompts from evaluation_metadata.jsonl and groups them by tag.
    Also stores the line number for each prompt.
    
    Args:
        input_file (str): Path to the input JSONL file (evaluation_metadata.jsonl).
        output_file (str): Path to the output JSON file (filtered_prompts.json).
    """
    grouped_prompts = defaultdict(list)
    
    # Read the JSONL file line by line
    with open(input_file, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            tag = data["tag"]
            prompt = data["prompt"]
            line_number = idx + 1  # Store the line number
            grouped_prompts[tag].append({"prompt": prompt, "line_number": line_number})  # Group prompts by tag
    
    # Save the grouped prompts to a JSON file
    with open(output_file, "w") as f:
        json.dump(grouped_prompts, f, indent=4)
    
    print(f"✅ Extracted prompts grouped by tag and saved to {output_file}")

if __name__ == "__main__":
    input_file = "evaluation_metadata.jsonl"
    output_file = "filtered_prompts.json"
    extract_prompts(input_file, output_file)