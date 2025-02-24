import os
import torch
from functools import partial
from diffusers import StableDiffusionXLPipeline
from torchvision import transforms
from PIL import Image
from qwen_integration import get_refined_prompt

# Configuration
SEED = 42
SAVE_STEPS = {9, 24, 74, 99}  # Steps to save intermediates
BASE_DIR = "./imgs"
os.makedirs(BASE_DIR, exist_ok=True)
tag = "position"

def parse_qwen_output(full_output):
    """Parses Qwen output to extract decision (True/False) and refined prompt."""
    decision, refined_prompt = None, None

    for line in full_output.split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().strip('"')
            print(decision)
        elif line.startswith("REFINED PROMPT:"):
            refined_prompt = line.replace("REFINED PROMPT:", "").strip().strip('"')

    return decision, refined_prompt or "None"  # Default if no refined prompt

prompt = "A photo of a cat on left and dog on right."

# Initialize pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")
pipe.enable_model_cpu_offload()  # Offloads unused components to CPU

# Helper function to convert latents to images
def latents_to_pil(pipe, latents):
    with torch.no_grad():
        original_dtype = pipe.vae.dtype
        pipe.vae.to(torch.float32)
        images = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        pipe.vae.to(original_dtype)
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray((img * 255).astype("uint8")) for img in images]

# Callback function
def generation_callback(pipe, step_index, timestep, callback_kwargs, 
                        swap_step=None, new_prompt=None, save_steps=SAVE_STEPS,
                        img_dir=None, latent_dir=None):
    latents = callback_kwargs.get("latents")
    if latents is None:
        return callback_kwargs

    # Save intermediates at specified steps
    if step_index in save_steps:
        torch.save(latents, os.path.join(latent_dir, f"step_{step_index}.pt"))
        pil_images = latents_to_pil(pipe, latents)
        pil_images[0].save(os.path.join(img_dir, f"step_{step_index}.png"))

    # Handle prompt swapping
    if swap_step is not None and step_index == swap_step and new_prompt:
        print(f"\n=== SWAPPING PROMPT AT STEP {swap_step} ===")

        cfg_enabled = pipe.do_classifier_free_guidance
        prompt_embeds = pipe.encode_prompt(
            new_prompt,
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

# Experiment loop
previous_experiment = None
refined_prompt = None  # Will be updated after first experiment

for i, exp in enumerate([
    {"name": "no_swap", "swap_step": None},
    {"name": "swap_at_24", "swap_step": 24},
    {"name": "swap_at_9", "swap_step": 9},
    {"name": "swap_at_0", "swap_step": 0}
]):
    print(f"\n🚀 Starting experiment: {exp['name']}")

    # Setup experiment directories
    exp_dir = os.path.join(BASE_DIR, exp["name"])
    img_dir = os.path.join(exp_dir, "images")
    latent_dir = os.path.join(exp_dir, "latents")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    # Reset generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    # Prepare prompt: use refined prompt from previous experiment if available
    if i == 0:
        exp["original_prompt"] = prompt  # First experiment uses original prompt
        exp["new_prompt"] = None
    else:
        exp["original_prompt"] = prompt
        exp["new_prompt"] = refined_prompt  # Use refined prompt from previous step

    # Create customized callback
    callback_partial = partial(
        generation_callback,
        swap_step=exp["swap_step"],
        new_prompt=exp["new_prompt"],
        img_dir=img_dir,
        latent_dir=latent_dir
    )

    # Run generation
    pipe(
        prompt=exp["original_prompt"],
        num_inference_steps=100,
        callback_on_step_end=callback_partial,
        generator=generator
    )

    # Extract step 74 image for Qwen refinement (if not last experiment)
    if i < 3:  # No need to refine for the last experiment
        step_74_image_path = os.path.join(img_dir, "step_99.png")
        
        if os.path.exists(step_74_image_path):
            print(f"\n🔍 Passing {step_74_image_path} to Qwen for refinement...")
            full_output = get_refined_prompt(prompt, step_74_image_path, tag)
            _, refined_prompt = parse_qwen_output(full_output)
            print(f"\n✅ Refined Prompt for next experiment: {refined_prompt}")
        else:
            print(f"\n❌ Step 74 image missing for {exp['name']}, using previous refined prompt.")

print("\n✅ All experiments completed!")
