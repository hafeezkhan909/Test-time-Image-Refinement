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

def get_refined_prompt(original_prompt, latent_image_path, tag, prompt_history=None):
    """
    Uses Qwen2.5-VL to analyze if the image aligns with the original prompt
    and refine the prompt based on prompt history.
    
    Args:
        original_prompt (str): The initial user prompt.
        latent_image_path (str): Path to the saved latent image.
        tag (str): The tag associated with the prompt (e.g., "single_object").
        prompt_history (list, optional): List of previous prompt refinements.
    
    Returns:
        str: A refined prompt based on Qwen2.5-VL's analysis.
    """
    # Get the last used prompt (if available)
    last_prompt = original_prompt
    if prompt_history and len(prompt_history) > 0:
        last_prompt = prompt_history[-1]
    
    # Format prompt history if provided
    history_text = ""
    if prompt_history and len(prompt_history) > 0:
        history_text = "### Previous Prompt Refinements:\n"
        for i, prev_prompt in enumerate(prompt_history):
            history_text += f"- Refinement {i+1}: \"{prev_prompt}\"\n"
    
    # Add tag-specific logic here
    if tag == "single_object":
        # Example: Add specific instructions for single-object prompts
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{latent_image_path}"},
                    {"type": "text", "text": f"""
                    ### Evaluation Task:
                    You are an **Image Improvement Assistant**. Your job is to help make the image more aligned with the ORIGINAL prompt.

                    ### **Given Inputs:**  
                    1. **Original User Prompt:**  
                    - {original_prompt}  

                    2. **Last Used Prompt:**
                    - {last_prompt}

                    3. **Prompt History:**
                    {history_text}

                    4. **Current Image Analysis:**
                    - Look at the image and identify what aspects DIFFER from what the ORIGINAL prompt requested
                    - Analyze what essential elements from the ORIGINAL prompt are missing or incorrectly represented
                    - Ignore image quality issues like noise, blurriness, or artifacts
                    
                    ### **Your Task:**  
                    1. Create a NEW PROMPT that will help generate an image that better matches the ORIGINAL prompt
                    2. Your new prompt should be a modification of the last used prompt
                    3. Focus on fixing what's missing or incorrectly represented in the current image
                    4. The goal is to get progressively closer to fulfilling the ORIGINAL prompt
                    
                    ### **Decision Process:**
                    1. If the image ALREADY closely represents the ORIGINAL prompt:
                    
                    DECISION: "True"
                    REFINED PROMPT: "<An enhanced version of the last prompt that maintains alignment>"

                    2. If the image DOES NOT adequately represent the ORIGINAL prompt:
                    
                    DECISION: "False"
                    REFINED PROMPT: "<Your NEW prompt that addresses the specific misalignments>"

                    Follow this exact output format:
                    DECISION: "True" or "False"
                    REFINED PROMPT: "<Your new prompt here>"
                    """}
                ],
            }
        ]
    elif tag == "two_object":
        # Example: Add specific instructions for multi-object prompts
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{latent_image_path}"},
                    {"type": "text", "text": f"""
                    ### Evaluation Task:
                    You are an **Image Improvement Assistant**. Your job is to help make the image more aligned with the ORIGINAL prompt.

                    ### **Given Inputs:**  
                    1. **Original User Prompt:**  
                    - {original_prompt}  

                    2. **Last Used Prompt:**
                    - {last_prompt}

                    3. **Prompt History:**
                    {history_text}

                    4. **Current Image Analysis:**
                    - Look at the image and identify what aspects DIFFER from what the ORIGINAL prompt requested
                    - Analyze what essential elements from the ORIGINAL prompt are missing or incorrectly represented
                    - Ignore image quality issues like noise, blurriness, or artifacts
                    
                    ### **Your Task:**  
                    1. Create a NEW PROMPT that will help generate an image that better matches the ORIGINAL prompt
                    2. Your new prompt should be a modification of the last used prompt
                    3. Focus on fixing what's missing or incorrectly represented in the current image
                    4. The goal is to get progressively closer to fulfilling the ORIGINAL prompt
                    
                    ### **Decision Process:**
                    1. If the image ALREADY closely represents the ORIGINAL prompt:
                    
                    DECISION: "True"
                    REFINED PROMPT: "<An enhanced version of the last prompt that maintains alignment>"

                    2. If the image DOES NOT adequately represent the ORIGINAL prompt:
                    
                    DECISION: "False"
                    REFINED PROMPT: "<Your NEW prompt that addresses the specific misalignments>"

                    Follow this exact output format:
                    DECISION: "True" or "False"
                    REFINED PROMPT: "<Your new prompt here>"
                    """}
                ],
            }
        ]
    elif tag in ["position", "colors", "counting", "color_attr", "two_object2", "two_object3", 
                "single_object2", "single_object3", "single_object4", "position2"]:
        # Similar format for all other tags
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{latent_image_path}"},
                    {"type": "text", "text": f"""
                    ### Evaluation Task:
                    You are an **Image Improvement Assistant**. Your job is to help make the image more aligned with the ORIGINAL prompt.

                    ### **Given Inputs:**  
                    1. **Original User Prompt:**  
                    - {original_prompt}  

                    2. **Last Used Prompt:**
                    - {last_prompt}

                    3. **Prompt History:**
                    {history_text}

                    4. **Current Image Analysis:**
                    - Look at the image and identify what aspects DIFFER from what the ORIGINAL prompt requested
                    - Analyze what essential elements from the ORIGINAL prompt are missing or incorrectly represented
                    - Ignore image quality issues like noise, blurriness, or artifacts
                    
                    ### **Your Task:**  
                    1. Create a NEW PROMPT that will help generate an image that better matches the ORIGINAL prompt
                    2. Your new prompt should be a modification of the last used prompt
                    3. Focus on fixing what's missing or incorrectly represented in the current image
                    4. The goal is to get progressively closer to fulfilling the ORIGINAL prompt
                    
                    ### **Decision Process:**
                    1. If the image ALREADY closely represents the ORIGINAL prompt:
                    
                    DECISION: "True"
                    REFINED PROMPT: "<An enhanced version of the last prompt that maintains alignment>"

                    2. If the image DOES NOT adequately represent the ORIGINAL prompt:
                    
                    DECISION: "False"
                    REFINED PROMPT: "<Your NEW prompt that addresses the specific misalignments>"

                    Follow this exact output format:
                    DECISION: "True" or "False"
                    REFINED PROMPT: "<Your new prompt here>"
                    """}
                ],
            }
        ]
    else:
        # Default logic for other tags
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{latent_image_path}"},
                    {"type": "text", "text": f"""
                    ### Evaluation Task:
                    You are an **Image Improvement Assistant**. Your job is to help make the image more aligned with the ORIGINAL prompt.

                    ### **Given Inputs:**  
                    1. **Original User Prompt:**  
                    - {original_prompt}  

                    2. **Last Used Prompt:**
                    - {last_prompt}

                    3. **Prompt History:**
                    {history_text}

                    4. **Current Image Analysis:**
                    - Look at the image and identify what aspects DIFFER from what the ORIGINAL prompt requested
                    - Analyze what essential elements from the ORIGINAL prompt are missing or incorrectly represented
                    - Ignore image quality issues like noise, blurriness, or artifacts
                    
                    ### **Your Task:**  
                    1. Create a NEW PROMPT that will help generate an image that better matches the ORIGINAL prompt
                    2. Your new prompt should be a modification of the last used prompt
                    3. Focus on fixing what's missing or incorrectly represented in the current image
                    4. The goal is to get progressively closer to fulfilling the ORIGINAL prompt
                    
                    ### **Decision Process:**
                    1. If the image ALREADY closely represents the ORIGINAL prompt:
                    
                    DECISION: "True"
                    REFINED PROMPT: "<An enhanced version of the last prompt that maintains alignment>"

                    2. If the image DOES NOT adequately represent the ORIGINAL prompt:
                    
                    DECISION: "False"
                    REFINED PROMPT: "<Your NEW prompt that addresses the specific misalignments>"

                    Follow this exact output format:
                    DECISION: "True" or "False"
                    REFINED PROMPT: "<Your new prompt here>"
                    """}
                ],
            }
        ]

    # Process input (text + image) and generate refined prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,
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