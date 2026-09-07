import os
import torch
from tqdm.auto import tqdm
from diffusion.models import load_models
from diffusion.image_utils import decode_latents, save_image
from diffusion.config import DEVICE, HEIGHT, WIDTH, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, BATCH_SIZE

def refine_image(refined_prompt, init_latents, start_timestep=25, save_intermediate_steps=False,
                  refinement_step=None, adjusted_refinement_step=None, output_dir=None, prefix="",
                  model_version="", num_inference_steps=None, guidance_scale=None):
    """
    Refines an image starting from a latent state (e.g. at t=25) up to the final clean image.
    
    Args:
        refined_prompt (str): The refined text prompt to guide denoising.
        init_latents (torch.Tensor): The latent state from a previous run (e.g. saved at t=25).
        start_timestep (int): The timestep corresponding to the provided latent (default: 25).
        save_intermediate_steps (bool): Whether to save intermediate outputs during refinement.
        refinement_step (int): The step number to use when naming the saved intermediate image.
        adjusted_refinement_step (int): Index within the remaining (post-start_timestep) steps at
            which to save the intermediate image.
        output_dir (str): Directory to save refined images.
        prefix (str): Prefix for filenames to avoid overwriting.
        num_inference_steps (int): Total denoising steps in the schedule (default: config.NUM_INFERENCE_STEPS).
        guidance_scale (float): Classifier-free guidance scale (default: config.GUIDANCE_SCALE).
    
    Returns:
        final_image: The final refined image (after decoding).
        latents: The final latent state.
    """
    # Fall back to config.py defaults if not explicitly provided
    num_inference_steps = num_inference_steps or NUM_INFERENCE_STEPS
    guidance_scale = guidance_scale or GUIDANCE_SCALE

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
    scheduler.set_timesteps(num_inference_steps)
    
    latents = init_latents
    # Find the index in the scheduler's timesteps corresponding to start_timestep.
    # (Assumes that the scheduler's timesteps include the integer value start_timestep.)
    start_idx = None
    for idx, t in enumerate(scheduler.timesteps):
        if idx == start_timestep:
            start_idx = idx + 1
            break
    if start_idx is None:
        raise ValueError(f"Start timestep {start_timestep} not found in scheduler.timesteps.")

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
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample (i.e. update the latents)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Optionally save intermediate refined images
        if save_intermediate_steps and step == adjusted_refinement_step:
            mapped_step = refinement_step
            print(f"Saving intermediate refined image at step {step} (saving as {mapped_step}) (timestep {t})")
            intermediate_image = decode_latents(latents, vae)
            image_path = os.path.join(output_dir, f"{prefix}step_{mapped_step}.png")  # Save with new step name
            save_image(intermediate_image, image_path)

    # Decode and save the final refined image (corresponds to the final timestep)
    final_image = decode_latents(latents, vae)
    final_image_path = os.path.join(output_dir, f"final_{prefix}.png")  # Add prefix
    save_image(final_image, final_image_path)

    return final_image, latents