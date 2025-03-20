import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from PIL import Image
import argparse

# Load Qwen2.5-VL model with memory optimizations
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# Load processor with reduced visual tokens for lower memory usage
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

# === X-IQE PROMPTS ===
FIDELITY_PROMPT = """
You are my assistant to evaluate the image quality. Briefly describe (within 50 words) the type (e.g., photo, painting) and content of this image, and analyze whether this image meets the following conditions of an AI-generated image (within 30 words per point).

1. Imperfect details: distorted, blurry, or irrational faces, limbs, fingers, objects, or texts.
2. Improper composition: some misplaced object relationships.
3. Strange colors: overly bright, saturated colors.
4. Artificial look: looks like a real image but has an unclear rendering or other artificial look.

Provide your analysis in JSON format with the following keys:  
- **Image description**  
- **Imperfect details**  
- **Improper composition**  
- **Strange colors**  
- **Artificial look**  
- **Fidelity (e.g., 6/10)**  
"""

ALIGNMENT_PROMPT_TEMPLATE = """
According to the image and your previous description, how well does the image align with the following description?

**Prompt used for generating this image:**  
{prompt}

Scoring Criteria:  
1 = not match at all  
2 = has significant discrepancies  
3 = has several minor discrepancies  
4 = has a few minor discrepancies  
5 = matches exactly  

Provide your analysis in JSON format with the following keys:  
- **Alignment analysis** (within 100 words)  
- **Alignment score (e.g., 4/5)**  
"""

AESTHETIC_PROMPT = """
Briefly analyze the aesthetic elements of this image (each item within 20 words) and score its aesthetics. The scoring criteria for each item are as follows.

0-1 = Extremely bad  
2-3 = Poor quality  
4 = Below average  
5 = Average  
6 = Above average  
7-8 = Good  
9 = Excellent  
10 = Wonderful  

Provide your analysis in JSON format with the following keys:  
- **Color harmony**  
- **Color brightness**  
- **Color saturation**  
- **Composition**  
- **Perspective**  
- **Light and shadow**  
- **Detailed expression**  
- **Vivid posture**  
- **Visual impact**  
- **Overall aesthetic score (e.g., 6/10)**  
"""

import re

def extract_json(text):
    """Extracts and cleans JSON from LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)  # Find first JSON block
    if match:
        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON, returning raw text.")
            return {"error": "Invalid JSON", "raw_output": text}
    return {"error": "No JSON detected", "raw_output": text}

def analyze_image(image_path, prompt):
    """Runs X-IQE evaluation using Qwen2.5-VL and extracts JSON."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process input (text + image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Decode output
    generated_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract valid JSON
    return extract_json(generated_response)

def evaluate_xiqe(image_folder, prompt_file, output_file):
    """Runs X-IQE evaluation on all images in a folder."""
    results = {}

    # Load prompts for alignment evaluation
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.jpg', '.png', '.jpeg', '.webp')):
            img_path = os.path.join(image_folder, img_name)
            print(f"\n🔍 Evaluating {img_name}...")

            # Extract prompt used to generate this image
            image_prompt = prompts.get(img_name, "No prompt available")

            fidelity = analyze_image(img_path, FIDELITY_PROMPT)
            alignment = analyze_image(img_path, ALIGNMENT_PROMPT_TEMPLATE.format(prompt=image_prompt))
            aesthetics = analyze_image(img_path, AESTHETIC_PROMPT)

            results[img_name] = {
                "Fidelity": fidelity,
                "Alignment": alignment,
                "Aesthetics": aesthetics
            }

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ X-IQE evaluation complete! Results saved in {output_file}.")

# === MAIN FUNCTION TO RUN THE EVALUATION ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run X-IQE Evaluation")
    parser.add_argument("--image_folder", required=True, type=str, help="Path to the folder containing images")
    parser.add_argument("--prompt_json", required=True, type=str, help="Path to the prompt JSON file")
    parser.add_argument("--output_json", required=True, type=str, help="Path to save the X-IQE results JSON file")
    args = parser.parse_args()

    evaluate_xiqe(args.image_folder, args.prompt_json, args.output_json)