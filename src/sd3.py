import os
import json
import torch
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline # For SD1.5
from diffusers import StableDiffusion3Pipeline
from qwen_integration import get_refined_prompt
from PIL import Image
import shutil
# Ready for SD3 run
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default="filtered_prompts.json")
    parser.add_argument("--output_dir", type=str, default="simple_outputs")
    return parser.parse_args()

def load_prompts(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def parse_qwen_output(full_output):
    decision = None
    refined_prompt = None
    for line in full_output.split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().strip('"')
        elif line.startswith("REFINED PROMPT:"):
            refined_prompt = line.replace("REFINED PROMPT:", "").strip().strip('"')
    return decision, refined_prompt if refined_prompt else "None"

def main():
    args = parse_args()
    prompts_by_tag = load_prompts(args.prompts_file)

    # ✅ Load Stable Diffusion 3
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     torch_dtype=torch.float16
    # ).to("cuda")

    for tag, prompts in prompts_by_tag.items():
        print(f"\n🔹 Processing tag: {tag}")
        tag_dir = os.path.join(args.output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        for prompt_data in prompts:
            prompt = prompt_data["prompt"]
            line_number = prompt_data["line_number"]
            print(f"\n Processing Prompt {line_number} for tag {tag}:\n{prompt}")

            prompt_id = f"prompt_{line_number:03d}"
            prompt_dir = os.path.join(tag_dir, prompt_id)
            os.makedirs(prompt_dir, exist_ok=True)

            # Step 1: Initial image
            image = pipe(prompt, negative_prompt="", num_inference_steps=100, guidance_scale=7.5).images[0]
            image_path = os.path.join(prompt_dir, "final_image.png")
            image.save(image_path)
            print(f"✅ Saved initial image: {image_path}")

            # ========================== #
            #    Refinement Loop
            # ========================== #
            prompt_history = []  # Track prompt history
            current_prompt = prompt
            for step in range(1, 4):
                full_output = get_refined_prompt(current_prompt, image_path, tag, prompt_history)
                decision, refined_prompt = parse_qwen_output(full_output)

                with open(os.path.join(prompt_dir, f"step_{step}_qwen.txt"), "w") as f:
                    f.write(full_output)

                print(f"Refining further: Refined Prompt for Step {step}: {refined_prompt}")
                # Add current refinement to history
                prompt_history.append(refined_prompt)
                
                if step == 1:
                    if decision == "True":
                        print(f"✅ Early stopping at {step} iter, image matches the prompt.")
                        es_final_path = os.path.join(prompt_dir, f"ES{step}_final_image.png")
                        shutil.copy(os.path.join(prompt_dir, "final_image.png"), es_final_path)
                else:
                    if decision == "True":
                        print(f"✅ Early stopping at {step} iter, image matches the prompt.")
                        es_final_path = os.path.join(prompt_dir, f"ES{step}_final_image.png")
                        prev_final_path = os.path.join(prompt_dir, f"step_{step-1}.png")
                        shutil.copy(prev_final_path, es_final_path)

                image = pipe(refined_prompt, negative_prompt="", num_inference_steps=100, guidance_scale=7.5).images[0]
                image_path = os.path.join(prompt_dir, f"step_{step}.png")
                image.save(image_path)
                print(f"✅ Step {step}: Saved image for refined prompt: {refined_prompt}")

                current_prompt = refined_prompt

    print("\n✅ All prompts processed successfully!")

if __name__ == "__main__":
    main()


