import os
import json
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from qwen_integration import get_refined_prompt

def parse_qwen_output(full_output):
    """Parses Qwen output to extract decision (True/False) and refined prompt."""
    decision = None
    refined_prompt = None
    for line in full_output.split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().strip('"')
        elif line.startswith("REFINED PROMPT:"):
            refined_prompt = line.replace("REFINED PROMPT:", "").strip().strip('"')
    return decision, refined_prompt

# Load prompts
with open("filtered_prompts.json", "r") as f:
    prompt_data = json.load(f)

SEED = 42
swap_steps = [None, 24, 9, 0]  # Order of swaps

def latents_to_pil(pipe, latents):
    with torch.no_grad():
        original_dtype = pipe.vae.dtype
        pipe.vae.to(torch.float32)
        images = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        pipe.vae.to(original_dtype)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray((img * 255).astype("uint8")) for img in images]

def create_callback(pipe, swap_step, output_dir, swap_prompt=None):
    def callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        save_steps = [74, 99]
        
        if step_index in save_steps:
            unique_name = f"swap_at_{swap_step}_step_{step_index}" if swap_step is not None else f"swap_at_None_step_{step_index}"
            image_path = os.path.join(output_dir, f"{unique_name}.png")
            latent_path = os.path.join(output_dir, f"{unique_name}.pt")
            
            pil_images = latents_to_pil(pipe, latents)
            pil_images[0].save(image_path)
            torch.save(latents, latent_path)
            print(f"\n🔍 Saved output: {image_path}")

        if step_index == swap_step and swap_prompt:
            print(f"\n=== SWAPPING PROMPT AT STEP {swap_step} ===")
            print(f"🔄 New Prompt: {swap_prompt}")
            
            cfg_enabled = pipe.do_classifier_free_guidance
            prompt_embeds = pipe.encode_prompt(
                swap_prompt,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=cfg_enabled,
                negative_prompt=""
            )
            
            if cfg_enabled:
                final_embeds = torch.cat([prompt_embeds[1], prompt_embeds[0]], dim=0)
                final_pooled = torch.cat([prompt_embeds[3], prompt_embeds[2]], dim=0)
            else:
                final_embeds = prompt_embeds[0]
                final_pooled = prompt_embeds[2]

            callback_kwargs["prompt_embeds"] = final_embeds
            callback_kwargs["pooled_prompt_embeds"] = final_pooled

        return callback_kwargs
    return callback

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")
pipe.enable_model_cpu_offload()

for tag, prompts in prompt_data.items():
    for prompt_info in prompts:
        original_prompt = prompt_info["prompt"]
        line_number = prompt_info["line_number"]
        prompt_id = f"prompt_{line_number:03d}"
        output_dir = os.path.join("./imgs/dynamic_refinements", tag, prompt_id)
        os.makedirs(output_dir, exist_ok=True)
        
        current_refined_prompt = None

        for idx, swap_step in enumerate(swap_steps):
            print(f"\n🚀 Processing: {tag}/{prompt_id} | Swap at {swap_step}")
            
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            pipe(
                prompt=original_prompt,
                num_inference_steps=100,
                callback_on_step_end=create_callback(
                    pipe, 
                    swap_step,
                    output_dir,
                    swap_prompt=current_refined_prompt if current_refined_prompt else None
                ),
                generator=generator
            )

            if swap_step == 0:
                old_path = os.path.join(output_dir, "swap_at_0_step_99.png")
                new_path = os.path.join(output_dir, "final_final_image.png")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"🔄 Renamed {old_path} to {new_path}")

            if idx < len(swap_steps) - 1:
                step_74_image_path = os.path.join(output_dir, f"swap_at_{swap_step}_step_99.png")
                full_output = get_refined_prompt(original_prompt, step_74_image_path, tag)
                decision, current_refined_prompt = parse_qwen_output(full_output)
                print(f"\n🔄 Qwen refined prompt: {current_refined_prompt}")
                
                if decision == "True":
                    old_path = os.path.join(output_dir, f"swap_at_{swap_step}_step_99.png")
                    new_path = os.path.join(output_dir, f"ES_{idx+1}_final_image.png")
                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)
                        print(f"🔄 Renamed {old_path} to {new_path}")
