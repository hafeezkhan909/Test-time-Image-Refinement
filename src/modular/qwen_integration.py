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

def get_refined_prompt(original_prompt, latent_image_path, tag):
    """
    Uses Qwen2.5-VL to analyze the latent image and refine the prompt.
    
    Args:
        original_prompt (str): The initial user prompt.
        latent_image_path (str): Path to the saved latent image.
        tag (str): The tag associated with the prompt (e.g., "single_object").
    
    Returns:
        str: A refined prompt based on Qwen2.5-VL's analysis.
    """
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a potted plant."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The potted plant is clearly visible in the frame.  
                    - **Correct Shape & Structure**: The plant has distinct leaves and a well-defined pot.  
                    - **Natural Appearance**: The leaves are detailed, and the pot has realistic texture.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-defined photo of a potted plant with vibrant green leaves and a sturdy ceramic pot. The plant should have rich foliage, with individual leaves clearly visible and detailed. The pot should have a smooth, textured surface that complements the plant. The background should be neutral and softly blurred to maintain focus on the plant, ensuring a clean and aesthetically balanced composition."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a baseball bat."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The image appears to have an elongated object, but it is not clearly distinguishable as a baseball bat.  
                    - **Shape Issue**: The form is slightly irregular, making it unclear whether it is a bat or another cylindrical object.  
                    - **Texture Issue**: The wooden texture is missing, and the bat appears too smooth, resembling a metal rod.  
                    - **Background Clarity**: The bat blends into the background, making it difficult to distinguish.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a classic wooden baseball bat with a smooth, polished surface and visible wood grain texture. The bat should have a rounded barrel and a tapered handle with a firm grip. The image should be well-lit to highlight the shape and details of the bat. The background should be neutral and softly blurred to emphasize the bat as the focal point of the composition."*  
                    
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
    elif tag == "two_object":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "two_object2":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "two_object3":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "single_object2":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "single_object3":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "single_object4":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 2:**  
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a toothbrush and a snowboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The toothbrush is clearly visible, but the snowboard is faint and partially obscured.  
                    - **Size Discrepancy**: The snowboard appears disproportionately small compared to the toothbrush.  
                    - **Object Positioning**: The toothbrush is centered, but the snowboard is placed awkwardly in the background, making it difficult to recognize.  
                    - **Texture Issue**: The snowboard lacks the glossy surface and defined edges typical of a snowboard.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a toothbrush and a snowboard, both clearly visible and well-defined. The toothbrush should have a clean, ergonomic design with visible bristles and a smooth plastic handle. The snowboard should be large, with a sleek, glossy surface and distinctive graphics or branding. The two objects should be positioned naturally, with the toothbrush placed in the foreground and the snowboard fully visible in the background, ensuring proper scale and separation. The background should be neutral to avoid distraction while keeping both objects in sharp focus."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a horse and a computer keyboard."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the horse and the keyboard are clearly visible.  
                    - **Size & Perspective**: The horse appears large and well-proportioned, while the keyboard maintains a realistic size.  
                    - **Detail Accuracy**: The horse has a well-defined mane, muscular structure, and natural fur texture. The keyboard has visible keys with realistic spacing.  
                    - **Positioning**: The horse is placed in the background, and the keyboard is in the foreground, ensuring both objects remain distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-composed photo featuring a fully visible horse in the background and a computer keyboard in the foreground. The horse should be detailed, with a flowing mane, visible musculature, and natural fur texture. The keyboard should be modern, with clearly defined keys and a structured layout. The scene should have a neutral backdrop to enhance clarity and avoid blending, ensuring both objects are distinct while maintaining natural proportions and realistic lighting."*                      
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
    elif tag == "position2":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a dog to the right of a teddy bear."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the dog and the teddy bear are clearly visible.  
                    - **Positioning Accuracy**: The dog is correctly placed to the right of the teddy bear, maintaining the intended spatial relationship.  
                    - **Detail & Clarity**: The dog has well-defined fur texture, expressive eyes, and a natural pose. The teddy bear has a soft, plush appearance with visible stitching and fabric texture.  
                    - **Background & Framing**: The scene is well-balanced, ensuring both objects remain clearly visible without overlap.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-framed photo of a dog positioned to the right of a teddy bear. The dog should have natural fur texture, expressive eyes, and a relaxed sitting or standing posture. The teddy bear should appear soft and plush, with detailed stitching and a well-defined fabric surface. The spatial arrangement should be clear, ensuring the dog is distinctly positioned to the right of the teddy bear, with proper separation and visibility. The background should be softly blurred to maintain focus on both subjects."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a bus below a toothbrush."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the bus and toothbrush are visible.  
                    - **Incorrect Positioning**: The bus is placed next to or above the toothbrush rather than below it.  
                    - **Size & Proportion**: The toothbrush appears disproportionately large compared to the bus, making the spatial relationship unclear.  
                    - **Scene Composition**: The objects appear disconnected rather than forming a cohesive arrangement.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A well-structured photo of a bus positioned clearly below a toothbrush. The bus should be realistically sized, maintaining its natural proportions and details such as windows, headlights, and doors. The toothbrush should be placed above the bus, appearing appropriately scaled with visible bristles and an ergonomic handle. The background should provide a neutral setting, ensuring the spatial positioning remains clear and easily distinguishable."*                                          
                                        
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
    elif tag == "counting":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of two ovens."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Count Issue**: Only one oven is clearly visible in the image. The second oven is either missing or faintly present in the background.  
                    - **Size & Proportion**: The visible oven appears accurate in size and design, but the second oven is either too small or not clearly defined.  
                    - **Positioning Issue**: The ovens are not placed in a natural arrangement; they appear overlapping rather than side by side.  
                    - **Detail & Clarity**: The control panel, doors, and handles of the visible oven are distinct, but the second oven lacks clarity.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of two full-sized ovens, positioned side by side in a well-lit environment. Each oven should be distinct, with clear doors, control panels, and visible handles. The surfaces should reflect light naturally, showcasing the metallic or matte finish. The composition should ensure that both ovens are completely visible, well-proportioned, and not overlapping. The background should remain neutral to maintain focus on the appliances."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of four dogs."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Correct Object Count**: Four dogs are fully present in the image.  
                    - **Proper Spacing & Layout**: The dogs are evenly spaced and do not overlap unnaturally.  
                    - **Detail Accuracy**: Each dog has well-defined facial features, fur textures, and natural body posture.  
                    - **Lighting & Focus**: The image has balanced lighting, ensuring all four dogs remain clearly visible and distinguishable.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-balanced photo featuring four distinct dogs, each clearly visible and well-proportioned. The dogs should have natural fur textures, expressive eyes, and detailed body features. They should be positioned evenly across the frame, ensuring none are obstructed or overlapping. The background should be softly blurred, keeping the focus on all four dogs while maintaining a natural and aesthetically pleasing composition."*                      ---
                    
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
    elif tag == "colors":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a blue cow."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Color Issue**: The cow appears in a natural brown or black shade rather than blue.  
                    - **Object Accuracy**: The cow is well-formed, with clear features such as ears, horns, and a tail.  
                    - **Texture & Realism**: The fur texture is realistic, but the intended color modification is not applied.  
                    - **Background & Composition**: The cow is well-positioned in the frame, but the image does not reflect the requested color modification.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A detailed photo of a blue-colored cow standing in an open field. The cow’s fur should be a rich, vibrant blue, evenly distributed across its body while maintaining a natural fur texture. Its facial features, ears, and tail should be clearly visible, ensuring a realistic representation despite the color modification. The background should be a softly blurred pasture, ensuring the focus remains on the uniquely colored cow."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a purple hair drier."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Correct Object Presence**: The hair drier is fully visible and identifiable.  
                    - **Color Accuracy**: The hair drier is a vivid purple, accurately reflecting the prompt.  
                    - **Detail & Texture**: The plastic surface has a smooth, glossy finish, with visible vents and buttons.  
                    - **Lighting & Focus**: The object is well-lit, ensuring all design details are clearly visible.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A high-quality photo of a sleek, modern purple hair drier with a glossy finish. The hair drier should have clearly visible vents, buttons, and ergonomic contours. The purple color should be rich and evenly distributed, ensuring it stands out while maintaining a realistic material appearance. The background should be neutral, softly blurred, and non-distracting to keep the focus on the hair drier."*                      
                                        
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
    elif tag == "position":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a dog to the right of a teddy bear."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the dog and the teddy bear are clearly visible.  
                    - **Positioning Accuracy**: The dog is correctly placed to the right of the teddy bear, maintaining the intended spatial relationship.  
                    - **Detail & Clarity**: The dog has well-defined fur texture, expressive eyes, and a natural pose. The teddy bear has a soft, plush appearance with visible stitching and fabric texture.  
                    - **Background & Framing**: The scene is well-balanced, ensuring both objects remain clearly visible without overlap.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A well-framed photo of a dog positioned to the right of a teddy bear. The dog should have natural fur texture, expressive eyes, and a relaxed sitting or standing posture. The teddy bear should appear soft and plush, with detailed stitching and a well-defined fabric surface. The spatial arrangement should be clear, ensuring the dog is distinctly positioned to the right of the teddy bear, with proper separation and visibility. The background should be softly blurred to maintain focus on both subjects."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a bus below a toothbrush."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the bus and toothbrush are visible.  
                    - **Incorrect Positioning**: The bus is placed next to or above the toothbrush rather than below it.  
                    - **Size & Proportion**: The toothbrush appears disproportionately large compared to the bus, making the spatial relationship unclear.  
                    - **Scene Composition**: The objects appear disconnected rather than forming a cohesive arrangement.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A well-structured photo of a bus positioned clearly below a toothbrush. The bus should be realistically sized, maintaining its natural proportions and details such as windows, headlights, and doors. The toothbrush should be placed above the bus, appearing appropriately scaled with visible bristles and an ergonomic handle. The background should provide a neutral setting, ensuring the spatial positioning remains clear and easily distinguishable."*                                          
                                        
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
    elif tag == "color_attr":
        # Example: Add specific instructions for multi-object prompts
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

                    ### **Example 3**  

                    #### **Original User Prompt:**  
                    *"A photo of a yellow skateboard and an orange computer mouse."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: Both the skateboard and the computer mouse are clearly visible.  
                    - **Color Accuracy**: The skateboard is vibrant yellow, and the computer mouse is distinctly orange.  
                    - **Object Detail**: The skateboard has a well-defined deck, wheels, and grip tape, while the mouse has a smooth, ergonomic shape with visible buttons.  
                    - **Positioning & Framing**: The objects are placed side by side in a well-lit, balanced composition.  

                    DECISION: "True"  
                    REFINED PROMPT: *"A high-quality photo featuring a bright yellow skateboard and a vividly orange computer mouse. The skateboard should have a smooth, well-defined deck, sturdy wheels, and visible grip tape. The computer mouse should be a rich, warm orange with a sleek, ergonomic design and clearly visible buttons. Both objects should be positioned side by side, with proper lighting to ensure their colors remain bold and distinct. The background should be neutral to maintain focus on the objects."*  

                    ---

                    ### **Example 4**  

                    #### **Original User Prompt:**  
                    *"A photo of a white dining table and a red car."*  

                    **Analysis of Latent Image at Step 75:**  
                    - **Object Presence**: The red car is clearly visible, but the dining table appears in a different color or is faint.  
                    - **Color Issue**: The table appears light beige or gray instead of white.  
                    - **Detail & Texture**: The car has a well-defined body with clear reflections, but the table lacks strong highlights to emphasize its color.  
                    - **Scene Composition**: The car dominates the frame, while the table is either too small or positioned awkwardly.  

                    DECISION: "False"  
                    REFINED PROMPT: *"A high-quality photo of a pristine white dining table and a bold red car. The dining table should be bright white, with a smooth surface that reflects light naturally, ensuring it is distinctly white without gray or beige tones. The red car should have a vibrant, glossy finish with visible details such as headlights, doors, and reflections. Both objects should be proportionally balanced in the frame, ensuring the table is fully visible and clearly distinguishable from the background."*                                          
                                        
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
    else:
        # Default logic for other tags
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
                    ### **Decision Process:**  
                    1. If the **image represents the {original_prompt} prompt**, output:  
                       
                       DECISION: "True"
                       REFINED PROMPT: "<Your improved single prompt here>"

                    2. If the **image does not represent the prompt at all or has inconsistencies**, output:  
                       DECISION: "False"
                       REFINED PROMPT: "<Your improved single prompt here>"
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