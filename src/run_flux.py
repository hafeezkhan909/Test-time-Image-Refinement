import os
import json
import torch
from argparse import ArgumentParser
import shutil

# FLUX.1-dev is a gated checkpoint on Hugging Face. Before running this script,
# accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
# and authenticate locally, e.g. `huggingface-cli login` or set HF_TOKEN.

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default="filtered_prompts.json")
    parser.add_argument("--output_dir", type=str, default="new_outputs/geneval_batch_results_flux")
    parser.add_argument("--refinement_iterations", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--mllm", type=str, default="qwen", choices=["qwen", "gpt4o"])
    return parser.parse_args()

def load_refiner(mllm):
    if mllm == "qwen":
        from qwen_integration import get_refined_prompt
    elif mllm == "gpt4o":
        from aoai import get_refined_prompt
    else:
        raise ValueError(f"Unknown mllm choice: {mllm!r}")
    return get_refined_prompt

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

def check_image_exists(filepath):
    return os.path.exists(filepath)

def main():
    args = parse_args()
    get_refined_prompt = load_refiner(args.mllm)

    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    prompts_by_tag = load_prompts(args.prompts_file)

    restart_steps = [0] * args.refinement_iterations

    for tag, prompts in prompts_by_tag.items():
        print(f"\nProcessing tag: {tag}")
        tag_dir = os.path.join(args.output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        for prompt_data in prompts:
            prompt = prompt_data["prompt"]
            line_number = prompt_data["line_number"]
            print(f"\nProcessing Prompt {line_number} for tag {tag}:\n{prompt}")

            prompt_id = f"prompt_{line_number:03d}"
            prompt_dir = os.path.join(tag_dir, prompt_id)
            os.makedirs(prompt_dir, exist_ok=True)

            # Step 1: Initial image
            final_image_path = os.path.join(prompt_dir, "final_image.png")
            if check_image_exists(final_image_path):
                print(f"Initial image already exists for prompt {line_number}. Skipping generation.")
            else:
                image = pipe(
                    prompt=prompt, height=args.height, width=args.width,
                    num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale
                ).images[0]
                image.save(final_image_path)
                print(f"Saved initial image: {final_image_path}")

            # ========================== #
            #    Refinement Loop
            # ========================== #
            prompt_history = []  # Track prompt history

            for i, restart_step in enumerate(restart_steps):
                if i == 0:
                    prev_step_image = final_image_path
                else:
                    prev_restart_step = restart_steps[i - 1]
                    prev_step_image = os.path.join(prompt_dir, f"final_{i}_refined_{prev_restart_step}_.png")

                refined_prompt_path = os.path.join(prompt_dir, f"{i}_refined_prompt_{restart_step}.txt")
                if check_image_exists(refined_prompt_path):
                    print(f"Refined prompt for step {restart_step} already exists. Loading from file.")
                    with open(refined_prompt_path, "r") as f:
                        full_output = f.read()
                else:
                    full_output = get_refined_prompt(prompt, prev_step_image, tag, prompt_history)
                    with open(refined_prompt_path, "w") as f:
                        f.write(full_output)

                decision, refined_prompt = parse_qwen_output(full_output)
                print(f"Refining further: Refined Prompt for Step {restart_step}: {refined_prompt}")
                prompt_history.append(refined_prompt)

                if decision == "True":
                    print(f"Faithful image produced at iteration {i+1}, marked as ES{i+1} (refinement continues).")
                    es_final_path = os.path.join(prompt_dir, f"ES{i+1}_final_image.png")
                    if not check_image_exists(es_final_path):
                        shutil.copy(prev_step_image, es_final_path)

                refined_image_path = os.path.join(prompt_dir, f"final_{i+1}_refined_{restart_step}_.png")
                if check_image_exists(refined_image_path):
                    print(f"Refined image for step {restart_step} already exists. Skipping generation.")
                else:
                    image = pipe(
                        prompt=refined_prompt, height=args.height, width=args.width,
                        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale
                    ).images[0]
                    image.save(refined_image_path)
                    print(f"Step {i+1}: Saved image for refined prompt: {refined_prompt}")

    print("\nAll prompts processed successfully!")

if __name__ == "__main__":
    main()