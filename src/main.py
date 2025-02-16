from diffusion_pipeline import generate_image
import random

# Define prompt
PROMPT = "a photo of an orange"

# Run image generation
generate_image(PROMPT, generator_seed=random.randint(0, 1000000), save_intermediate_steps=True)
# random.randint(0, 1000000)
# 0	"A realistic photo of a scene with a black and white panda and green bamboos."
# 1	"A realistic photo of a scene with a black horse and white fence."
# 2	"A realistic photo of a scene with a blue bird and yellow flower."
# 3	"A realistic photo of a scene with a blue chair and red table."
# 4	"A realistic photo of a scene with a gray curtain bird and purple sofa."
# 5	"A realistic photo of a scene with a gray elephant and brown tree."
# 6	"A realistic photo of a scene with a orange basketball and purple water bottle."
# 7	"A realistic photo of a scene with a orange book and blue pen."
# 8	"A realistic photo of a scene with a orange cat and blue ball."
# 9	"A realistic photo of a scene with a orange cat and red velvet sofa."
# 10	"A realistic photo of a scene with a pink rose and yellow butterfly."
# 11	"A realistic photo of a scene with a silver car and red stop sign"
# 12	"A realistic photo of a scene with a white rabbit and green leaf."
# 13	"A realistic photo of a scene with a white sheep and gray rock."
# 14	"A realistic photo of a scene with a white swan and dark blue lake."
# 15	"A realistic photo of a scene with a brown teddy and black hat."