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
### Evaluation Task:
You are an **Image Refinement Assistant**. Your job is to check for **image-prompt correctness** and refine the prompt ONLY. 

### **Given Inputs:**  
1. **Original User Prompt:**  
   - {original_prompt}  

2. **Look at what is within the latent image:**  
   - <You have to Describe what the generated image looks like>  
   - <List any specific issues, inconsistencies, or missing details with respect to the {original_prompt}>   

- **Do not check for image quality (e.g., sharpness, noise, lighting artifacts).**  
- **Ignore visual noise or distortions during evaluation.**  
- **Only assess whether the image correctly represents the original prompt.**   

---
### **Example 1**  

#### **Original User Prompt:**  
*"A fluffy gray rabbit with long ears wearing a tiny blue scarf."*  

**Analysis of Latent Image at Step 75:**
- Scarf Issue: The tiny blue scarf is not clearly visible or might be missing entirely.
- Fur Detail: The rabbit's fluffy fur is prominent, but it lacks clarity and fine detail, appearing overly textured or noisy.
- Background: The backdrop is a plain blue-gray color with minimal variation, which feels flat and unengaging.
- Rabbit Clarity: The rabbit’s form is discernible but slightly distorted, especially around the ears and face.

DECISION: "False"  
REFINED PROMPT: *"A highly detailed and fluffy gray rabbit with long, upright ears wearing a tiny, vibrant blue scarf wrapped around its neck. The rabbit should have soft, realistic fur texture and clear, expressive facial features. The background should be a softly blurred gradient of blue and gray tones, creating a serene atmosphere that highlights the rabbit as the focal point."*              

---

### **Example 2**  

#### **Original User Prompt:**  
*"An angry white dog next to a cute orange cat on a grassy hill at sunset."*  

**Analysis of Latent Image at Step 75:**
- The "angry white dog" is faintly discernible but lacks clear definition or features. It appears to blend into the background.
- The "cute orange cat" is indistinct, with no visible form or features, and might not be present at all.
- The grassy hill is visible but lacks texture and detail.
- The sunset lighting is absent, and the colors seem scattered without a clear gradient or sunset tones.
- Overall, the image lacks clarity, structure, and the contrast needed to align with the original prompt.

DECISION: "False"
REFINED PROMPT: *"An angry white dog with sharp features, standing next to a cute orange cat with large, expressive eyes on a textured grassy hill. The scene is illuminated by a vibrant sunset, with warm orange and pink hues filling the sky. The hill should have visible blades of grass, and the subjects should be sharply detailed with realistic textures."*  
---

### **Example 3:**

#### **Original User Prompt:**  
*"A photo of three giraffes."*  

**Analysis of Latent Image at Step 75:**  
- **Correct Object Count**: Three giraffes are visible in the frame.  
- **Giraffe Proportions**: All three giraffes appear natural in size and shape.  
 
DECISION: "True" 
REFINED PROMPT: *"A natural photo of three giraffes standing in an open grassland. The giraffes should be clearly visible with long necks and distinctive fur patterns. They should be positioned naturally, ensuring all three are fully within the frame and distinguishable from each other. The background should be soft and unobtrusive, keeping the focus on the giraffes."*  

---

### **Example 4:**  
#### **Original User Prompt:**  
*"A photo of four handbags."*  

**Analysis of Latent Image at Step 75:**  
- **Correct Object Count**: There are **exactly four handbags** visible in the image.  
- **Handbag Shape & Features**: Each handbag has clearly defined straps, zippers, or clasps, making them identifiable.  
- **Distinct Separation**: The handbags are positioned separately and do not merge into a single indistinct shape.  
- **Balanced Composition**: The handbags are evenly arranged in the frame, ensuring they are all fully visible.  
- **Ignored Factors**: Minor texture inconsistencies, lighting variations, or reflections **do not impact the evaluation**.  

DECISION: "True"
REFINED PROMPT: *"A well-lit, high-resolution photo featuring four distinct handbags arranged neatly on a flat surface. Each handbag has visible straps, metallic clasps, and a structured shape. The handbags should be evenly spaced, ensuring all four are fully visible without overlapping. The background should be neutral and unobtrusive to keep the focus on the handbags."*  
---

### **Decision Process:**  
1. If the **image represents the {original_prompt} prompt**, output:  
   
   DECISION: "True"
   REFINED PROMPT: "<Your improved single prompt here>"

2. If the **image does not represent the prompt at all or has inconsistencies**, output:  
   DECISION: "False"
   REFINED PROMPT: "<Your improved single prompt here>"

Note: Strictly follow the output format mentioned below.
DECISION: "True" or "False"
REFINED PROMPT: "<Your improved single prompt here>"
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
    test_prompt = "A photo of a cat."
    test_image_path = "outputs/final/final_image.png" 

    print("\n🔹 **Testing Qwen2.5-VL with Step 75 Latent Image...**")
    refined_prompt = get_refined_prompt(test_prompt, test_image_path)
    
    print("\n✅ **Refined Prompt:**")
    print(refined_prompt)