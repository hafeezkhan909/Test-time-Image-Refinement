import os
import torch
from tqdm.auto import tqdm
from diffusion.models import load_models
from diffusion.image_utils import decode_latents, save_image
from diffusion.config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE

def generate_image(prompt, generator_seed, save_intermediate_steps=False, output_dir=None, prefix="", model_version="",
                    restart_steps=None, refinement_step=None, num_inference_steps=None, guidance_scale=None,
                    height=None, width=None):
    """Runs Stable Diffusion pipeline and saves images at different timesteps.
       Also saves the latent state at every step listed in restart_steps.
    """
    # Fall back to config.py defaults if not explicitly provided
    num_inference_steps = num_inference_steps or NUM_INFERENCE_STEPS
    guidance_scale = guidance_scale or GUIDANCE_SCALE
    height = height or HEIGHT
    width = width or WIDTH
    restart_steps = restart_steps or []

    vae, tokenizer, text_encoder, unet, scheduler = load_models(DEVICE, model_version=model_version)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_output = text_encoder(text_input.input_ids.to(DEVICE))
        text_embeddings = text_output.last_hidden_state  # Use last_hidden_state

    # Classifier-free guidance: create unconditional embeddings
    uncond_input = tokenizer(
        [""] * BATCH_SIZE,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],  # Match prompt's length
        return_tensors="pt"
    )

    with torch.no_grad():
        uncond_output = text_encoder(uncond_input.input_ids.to(DEVICE))
        uncond_embeddings = uncond_output.last_hidden_state  # Use last_hidden_state

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Generate initial noise
    generator = torch.manual_seed(generator_seed)  # Same seed used for every call unless overridden via --seed
    latents = torch.randn((BATCH_SIZE, unet.config.in_channels, height // 8, width // 8), generator=generator).to(DEVICE)

    # Initialize scheduler
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma  # Scale latents
    latents_dict = {}  # Store intermediate latents instead of saving them to disk

    # Denoising Loop
    for step, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Save the latent state at every step listed in restart_steps
        if step in restart_steps:
            latents_dict[step] = latents.clone()
            print(f"Saved latent state at timestep {step} in memory")
        
        # Save intermediate images
        if save_intermediate_steps and step == refinement_step:
            print(f"Saving intermediate image at Step {step}, Timestep {t}")
            intermediate_image = decode_latents(latents, vae)
            image_path = os.path.join(output_dir, f"{prefix}step_{step}.png")  # Add prefix to filename
            save_image(intermediate_image, image_path)  # Pass the full path

    # Save final image
    print(f"Saving final image at Step {step}, Timestep {t}")
    print(f"The seed used is {generator_seed}.")

    # Save final image
    final_image_path = os.path.join(output_dir, f"final_{prefix}.png")  # Add prefix to filename
    save_image(decode_latents(latents, vae), final_image_path)

    return generator_seed, latents_dict