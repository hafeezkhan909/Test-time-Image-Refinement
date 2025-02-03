import os
import torch
from tqdm.auto import tqdm
from models import load_models
from image_utils import decode_latents, save_image
from config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE, GENERATOR_SEED

def generate_image(prompt, save_intermediate_steps=False):
    """Runs Stable Diffusion pipeline and saves images at different timesteps.
       Also saves the latent state when the timestep equals 25.
    """
    vae, tokenizer, text_encoder, unet, scheduler = load_models(DEVICE)
    
    # Tokenize prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]

    # Classifier-free guidance
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * BATCH_SIZE,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Generate initial noise
    generator = torch.manual_seed(GENERATOR_SEED)
    latents = torch.randn((BATCH_SIZE, unet.in_channels, HEIGHT // 8, WIDTH // 8), generator=generator).to(DEVICE)

    # Initialize scheduler
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    latents = latents * scheduler.init_noise_sigma  # Scale latents

    # Ensure output directories exist
    os.makedirs("outputs/intermediate", exist_ok=True)
    os.makedirs("outputs/final", exist_ok=True)
    os.makedirs("outputs/latents", exist_ok=True)

    # Select Intermediate Steps (for image saving)
    save_steps = [NUM_INFERENCE_STEPS // 10, NUM_INFERENCE_STEPS // 4, 
                  NUM_INFERENCE_STEPS // 2, 3 * NUM_INFERENCE_STEPS // 4]

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
        if step == 15:
            latent_path = f"outputs/latents/latent_t15_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 15 to {latent_path}")
        elif step == 25:
            latent_path = f"outputs/latents/latent_t25_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 25 to {latent_path}")
        elif step == 40:
            latent_path = f"outputs/latents/latent_t40_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 40 to {latent_path}")
        elif step == 50:
            latent_path = f"outputs/latents/latent_t50_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 50 to {latent_path}")
        elif step == 65:
            latent_path = f"outputs/latents/latent_t65_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 65 to {latent_path}")
        elif step == 75:
            latent_path = f"outputs/latents/latent_t75_{GENERATOR_SEED}.pt"
            torch.save(latents, latent_path)
            print(f"Saved latent state at timestep 75 to {latent_path}")

        # Save intermediate images
        if save_intermediate_steps and step in save_steps:
            print(f"Saving intermediate image at Step {step}, Timestep {t}")
            intermediate_image = decode_latents(latents, vae)
            save_image(intermediate_image, f"step_{step}.png", folder="outputs/intermediate")

    # Save final image
    print(f"Saving final image at Step {step}, Timestep {t}")
    final_image = decode_latents(latents, vae)
    save_image(final_image, "final_image.png", folder="outputs/final")

    return final_image
