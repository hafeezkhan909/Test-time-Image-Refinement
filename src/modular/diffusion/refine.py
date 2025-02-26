import os
import torch
from tqdm.auto import tqdm
from diffusion.models import load_models
from diffusion.image_utils import decode_latents, save_image
from diffusion.config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE

def refine_image(refined_prompt, init_latents, start_timestep=25, save_intermediate_steps=False, refinement_step=None, adjusted_refinement_step=None, output_dir=None, prefix="", model_version=""):
    """
    Refines an image starting from a latent state (e.g. at t=25) up to the final clean image.
    
    Args:
        refined_prompt (str): The refined text prompt to guide denoising.
        init_latents (torch.Tensor): The latent state from a previous run (e.g. saved at t=25).
        start_timestep (int): The timestep corresponding to the provided latent (default: 25).
        save_intermediate_steps (bool): Whether to save intermediate outputs during refinement.
        save_steps (set): Steps at which to save intermediate images.
        output_dir (str): Directory to save refined images.
        prefix (str): Prefix for filenames to avoid overwriting.
    
    Returns:
        final_image: The final refined image (after decoding).
        latents: The final latent state.
    """
    # Load all required models
    vae, tokenizer, text_encoder, unet, scheduler = load_models(DEVICE, model_version=model_version)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize the refined prompt
    text_input = tokenizer(
        refined_prompt,
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

    # (Re)initialize the scheduler timesteps
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    
    # Make sure the provided latent is on the correct device.
    # latents = init_latents.to(DEVICE)
    latents = init_latents
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
        # save_steps = {22, 43, 64}

    # save_steps = {38, 49, 57}

    save_step_mapping = {
    14: 15, 23: 25, 45: 50, 59: 65, 68: 75,  # Mapping for {14, 23, 45, 59, 68} → {15, 25, 50, 75}
    38: 50, 49: 65, 57: 75  # Mapping for {38, 49, 57} → {50, 65, 75}
    }

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
        if save_intermediate_steps and step == adjusted_refinement_step: # replace with save_steps to see multiple latent images at diff time steps
            
            mapped_step = refinement_step
            print(f"Saving intermediate refined image at step {step} (saving as {mapped_step}) (timestep {t})")
            intermediate_image = decode_latents(latents, vae)
            # latent_path = f"outputs/5_refined_latents/latent_t75.pt"
            # torch.save(latents, latent_path)
            image_path = os.path.join(output_dir, f"{prefix}step_{mapped_step}.png")  # Save with new step name
            save_image(intermediate_image, image_path)
            # save_image(intermediate_image, f"refined_step_{step}.png", folder="outputs/refined_intermediate")

    # Decode and save the final refined image (corresponds to t=100 or the final timestep)
    final_image = decode_latents(latents, vae)
    final_image_path = os.path.join(output_dir, f"final_{prefix}.png")  # Add prefix
    save_image(final_image, final_image_path)
    # save_image(final_image, "final_refined_image.png", folder="outputs/refined_final")

    return final_image, latents

# === Example usage in your main script ===

if __name__ == "__main__":
    # For example, assume that after running your first pipeline you saved the latent at t=25:
    # torch.save(latents, "outputs/latent_t25.pt")
    # init_latents = torch.load("outputs/5_latents/latent_t25_157618.pt")
    init_latents = torch.load("outputs/latents/latent_t10_488226.pt")

    # Define your refined prompt
    REFINED_PROMPT = "A warm and cozy library with rich, textured wooden bookshelves filled with books. The scene should have soft, focused lighting from a reading lamp, creating a warm glow without harsh shadows. Ensure there are no windows in the room, emphasizing an enclosed, intimate atmosphere. The composition should be balanced and detailed, with the wooden textures and cozy lighting as prominent features."

    # Run the refinement process from t=25 to t=100
    final_image, refined_latents = refine_image(REFINED_PROMPT, init_latents, start_timestep=10, save_intermediate_steps=True)
