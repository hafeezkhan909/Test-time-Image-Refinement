import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load Qwen2.5-VL model with memory optimizations
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Using flash attention 2
    device_map="auto"
)

# Load processor with reduced visual tokens for lower memory usage
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

# Function to generate refined prompt
def get_refined_prompt(original_prompt, latent_image_path="outputs/intermediate/step_75.png"):
    """
    Uses Qwen2.5-VL to analyze the latent image at step 75 and refine the prompt.

    Args:
        original_prompt (str): The initial user prompt.
        latent_image_path (str): Path to the saved latent image (default: step 75 image).

    Returns:
        str: A refined prompt based on Qwen2.5-VL's analysis.
    """

    # Construct the message for multimodal input (latent image + text)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{latent_image_path}"},
                {"type": "text", "text": f""" 
You are an **Image Refinement Assistant**, tasked with improving an image generation prompt based on mid-process observations in a diffusion model.

### **Context:**  
- We are at **step 75 out of 100** in the diffusion process.  
- At this point, **75% of noise has been removed**, and the latent image is partially formed.  
- The goal is to refine the **original user prompt** so that the final image at step 100 is better aligned with the user’s intent.  

---

### **Given Inputs:**  
1. **Original User Prompt:**  
   - {original_prompt}  

2. **Look at the image which is a Latent Image at Step 75:**  
   - <You have to Describe what the partially generated image looks like>  
   - <List any specific issues, inconsistencies, or missing details>  
   - (e.g., “The colors are too muted,” “The subject is off-center,” “We see extra artifacts,” “We want more vibrant lighting,” “We only see one cat instead of two,” etc.)  

---

### **Your Task:**  
1. **Analyze** the original prompt and compare it with the observations of the latent image.  
2. **Identify** discrepancies, missing elements, or visual artifacts that need correction.  
3. **Refine** the original prompt while staying true to the user’s intent. The refined prompt should:  
   - Maintain **scene and subject consistency** but clarify ambiguous details.  
   - Add **style, composition, or color guidance** if necessary.  
   - Optionally include **negative prompt terms** to remove unwanted elements or distortions.  

### **Example 1**  

#### **Original User Prompt:**  
*"A fluffy gray rabbit with long ears wearing a tiny blue scarf."*  

Analysis of Latent Image at Step 75:
- Scarf Issue: The tiny blue scarf is not clearly visible or might be missing entirely.
- Fur Detail: The rabbit's fluffy fur is prominent, but it lacks clarity and fine detail, appearing overly textured or noisy.
- Background: The backdrop is a plain blue-gray color with minimal variation, which feels flat and unengaging.
- Rabbit Clarity: The rabbit’s form is discernible but slightly distorted, especially around the ears and face.

#### **Refined Prompt:**  
*"A highly detailed and fluffy gray rabbit with long, upright ears wearing a tiny, vibrant blue scarf wrapped around its neck. The rabbit should have soft, realistic fur texture and clear, expressive facial features. The background should be a softly blurred gradient of blue and gray tones, creating a serene atmosphere that highlights the rabbit as the focal point."*  

### **Example 2**  

#### **Original User Prompt:**  
*"An angry white dog next to a cute orange cat on a grassy hill at sunset."*  

- The "angry white dog" is faintly discernible but lacks clear definition or features. It appears to blend into the background.
- The "cute orange cat" is indistinct, with no visible form or features, and might not be present at all.
- The grassy hill is visible but lacks texture and detail.
- The sunset lighting is absent, and the colors seem scattered without a clear gradient or sunset tones.
- Overall, the image lacks clarity, structure, and the contrast needed to align with the original prompt.

#### **Refined Prompt:**  
*"An angry white dog with sharp features, standing next to a cute orange cat with large, expressive eyes on a textured grassy hill. The scene is illuminated by a vibrant sunset, with warm orange and pink hues filling the sky. The hill should have visible blades of grass, and the subjects should be sharply detailed with realistic textures."*  
---

### **Output Format:**  
REFINED PROMPT: "<Your improved single prompt here>"

---

### **Constraints & Guidelines:**  
✅ **Preserve User Intent** – Do not introduce entirely new subjects or concepts unless needed for correction.  
✅ **Be Concise & Specific** – Ensure clarity while keeping the refined prompt succinct.  
✅ **Focus on Fixing Observed Issues** – Modify the prompt to correct image inconsistencies rather than making arbitrary changes.  
✅ **Enhance but Not Overwrite** – Improve detail, composition, and style without drastically altering the original vision.  

---
"""}
            ],
        }
    ]

    # Process input (text + image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate refined prompt
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim input portion and decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    refined_prompt = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return refined_prompt

# === MAIN FUNCTION FOR TESTING ===
if __name__ == "__main__":
    test_prompt = "A cozy library with wooden bookshelves and a reading lamp, but with no windows in sight."
    test_image_path = "outputs/intermediate/step_75.png" 

    print("\n🔹 **Testing Qwen2.5-VL with Step 75 Latent Image...**")
    refined_prompt = get_refined_prompt(test_prompt, test_image_path)
    
    print("\n✅ **Refined Prompt:**")
    print(refined_prompt)