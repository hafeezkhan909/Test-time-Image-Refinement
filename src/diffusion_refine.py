import torch
from tqdm.auto import tqdm
from models import load_models
from image_utils import decode_latents, save_image
from config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE

def refine_image(refined_prompt, init_latents, start_timestep=25, save_intermediate_steps=False):
    """
    Refines an image starting from a latent state (e.g. at t=25) up to the final clean image.
    
    Args:
        refined_prompt (str): The refined text prompt to guide denoising.
        init_latents (torch.Tensor): The latent state from a previous run (e.g. saved at t=25).
        start_timestep (int): The timestep corresponding to the provided latent (default: 25).
        save_intermediate_steps (bool): Whether to save intermediate outputs during refinement.
    
    Returns:
        final_image: The final refined image (after decoding).
        latents: The final latent state.
    """
    # Load all required models
    vae, tokenizer, text_encoder, unet, scheduler = load_models(DEVICE)

    # Tokenize the refined prompt
    text_input = tokenizer(
        refined_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]

    # Prepare classifier-free guidance embeddings
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * BATCH_SIZE,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]

    # Concatenate unconditional and text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # (Re)initialize the scheduler timesteps
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    
    # Make sure the provided latent is on the correct device.
    latents = init_latents.to(DEVICE)

    # Find the index in the scheduler’s timesteps corresponding to start_timestep.
    # (Assumes that the scheduler’s timesteps include the integer value start_timestep.)
    start_idx = None
    for idx, t in enumerate(scheduler.timesteps):
        if idx == start_timestep:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Start timestep {start_timestep} not found in scheduler.timesteps.")
    
    # Optionally define intermediate saving points. (Here we simply save the halfway point.)
    if save_intermediate_steps:
        remaining_steps = len(scheduler.timesteps[start_idx:])
        # For example, save at the middle of the remaining denoising process:
        save_steps = {25}
    
    # Continue the denoising loop starting from the provided latent state.
    remaining_timesteps = scheduler.timesteps[start_idx:]
    for step, t in enumerate(tqdm(remaining_timesteps, desc="Refining")):
        # Prepare latent input for classifier-free guidance (duplicate along batch dimension)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise using the UNet model
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform classifier-free guidance: split noise predictions and combine them
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample (i.e. update the latents)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Optionally save intermediate refined images
        if save_intermediate_steps and step in save_steps:
            print(f"Saving intermediate refined image at step {step} (timestep {t})")
            intermediate_image = decode_latents(latents, vae)
            # latent_path = f"outputs/5_refined_latents/latent_t75.pt"
            # torch.save(latents, latent_path)
            save_image(intermediate_image, f"refined_step_{step}.png", folder="outputs/refined_intermediate")

    # Decode and save the final refined image (corresponds to t=100 or the final timestep)
    final_image = decode_latents(latents, vae)
    save_image(final_image, "final_refined_image75.png", folder="outputs/refined_final")

    return final_image, latents

# === Example usage in your main script ===

if __name__ == "__main__":
    # For example, assume that after running your first pipeline you saved the latent at t=25:
    # torch.save(latents, "outputs/latent_t25.pt")
    # init_latents = torch.load("outputs/5_latents/latent_t25_157618.pt")
    init_latents = torch.load("outputs/5_refined_latents/latent_t75.pt")

    # Define your refined prompt
    REFINED_PROMPT = "A breathtaking fantasy landscape where the vibrant aurora borealis streaks vividly across the night sky in swirling waves of emerald green, violet, and electric blue, casting an ethereal glow over the tranquil world below. A herd of majestic reindeer, their thick, glistening fur reflecting the celestial hues, gracefully roams a vast, dew-kissed grassy meadow. Their elegantly branched antlers catch the soft silvery moonlight, creating a mesmerizing interplay of light and shadow. Nearby, a pristine, mirror-like lake stretches into the distance, flawlessly reflecting the dazzling aurora, its smooth surface rippling gently. In the far distance, a towering, snow-capped mountain rises with sharply defined, icy peaks, its slopes bathed in a soft lunar glow. The intricate textures of the reindeer’s fur, the lush grass, and the glacial mountain ridges are rendered in exquisite, ultra-detailed 8K resolution, enhancing the dreamlike atmosphere of this fantasy world."

    # Run the refinement process from t=25 to t=100
    final_image, refined_latents = refine_image(REFINED_PROMPT, init_latents, start_timestep=75, save_intermediate_steps=True)
