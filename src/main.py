from diffusion_pipeline import generate_image


# Define prompt
PROMPT = "Bathed in the soft glow of the setting sun, a lone traveler stands at the edge of a towering cliff, gazing over a mystical valley where golden rivers shimmer like liquid light. The wind gently tousles their flowing cloak as ethereal fireflies dance around, illuminating the twilight. In the distance, an ancient city with towering spires and glowing runes emerges from the mist, a place lost to time, waiting to reveal its secrets. Fantasy, 8K, highly detailed, cinematic lighting."

# Run image generation
generate_image(PROMPT, save_intermediate_steps=True)
