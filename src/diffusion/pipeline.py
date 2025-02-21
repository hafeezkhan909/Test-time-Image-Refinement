import os
import torch
from tqdm.auto import tqdm
from diffusion.models import load_models
from diffusion.image_utils import decode_latents, save_image
from diffusion.config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE

def generate_image(prompt, generator_seed, save_intermediate_steps=False, output_dir=None, prefix="", model_version=""):
    """Runs Stable Diffusion pipeline and saves images at different timesteps.
       Also saves the latent state when the timestep equals 25.
    """
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
    generator = torch.manual_seed(generator_seed)  # ✅ Each prompt gets a unique seed
    latents = torch.randn((BATCH_SIZE, unet.config.in_channels, HEIGHT // 8, WIDTH // 8), generator=generator).to(DEVICE)

    # Initialize scheduler
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latents = latents * scheduler.init_noise_sigma  # Scale latents
    latents_dict = {}  # Store intermediate latents instead of saving them to disk

    # Select Intermediate Steps (for image saving)
    save_steps = [NUM_INFERENCE_STEPS // 10, NUM_INFERENCE_STEPS * 3 // 20, NUM_INFERENCE_STEPS // 4, 
                  13 * NUM_INFERENCE_STEPS // 20, NUM_INFERENCE_STEPS // 2, 3 * NUM_INFERENCE_STEPS // 4]
    
    # Denoising Loop
    for step, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Save the latent state when the timestep equals 25
        if step in [10, 25]:
            latents_dict[step] = latents.clone()
            print(f"Saved latent state at timestep {step} in memory")
        
        # Save intermediate images
        if save_intermediate_steps and step in save_steps:
            print(f"Saving intermediate image at Step {step}, Timestep {t}")
            intermediate_image = decode_latents(latents, vae)
            image_path = os.path.join(output_dir, f"{prefix}step_{step}.png")  # Add prefix to filename
            save_image(intermediate_image, image_path)  # Pass the full path

    # Save final image
    print(f"Saving final image at Step {step}, Timestep {t}")
    print(f"The seed used is {generator_seed}.")

    # Save final image
    final_image_path = os.path.join(output_dir, f"{prefix}final_image.png")  # Add prefix to filename
    save_image(decode_latents(latents, vae), final_image_path)

    return generator_seed, latents_dict
