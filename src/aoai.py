import os
import base64
import mimetypes

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "https://dalle-exploration.openai.azure.com/")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2024-05-01-preview",
)

judge_model = os.environ.get("AOAI_JUDGE_MODEL", "gpt-4o")


def _encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_refined_prompt(original_prompt, image_path, tag, prompt_history=None):
    """
    Uses GPT-4o to judge whether the image aligns with the original prompt, and refine
    the prompt based on prompt history.

    Args:
        original_prompt (str): The initial user prompt.
        image_path (str): Path to the generated image to judge.
        tag (str): The tag associated with the prompt (e.g., "single_object").
        prompt_history (list, optional): List of previous prompt refinements.

    Returns:
        str: Raw model output containing a DECISION line and a REFINED PROMPT line.
    """
    last_prompt = original_prompt
    if prompt_history and len(prompt_history) > 0:
        last_prompt = prompt_history[-1]

    history_text = ""
    if prompt_history and len(prompt_history) > 0:
        history_text = "### Previous Prompt Refinements:\n"
        for i, prev_prompt in enumerate(prompt_history):
            history_text += f"- Refinement {i+1}: \"{prev_prompt}\"\n"

    if tag == "single_object":
        task_text = f"""
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
        """
    elif tag == "two_object":
        task_text = f"""
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
        """
    elif tag in ["position", "colors", "counting", "color_attr", "two_object2", "two_object3",
                 "single_object2", "single_object3", "single_object4", "position2"]:
        task_text = f"""
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
        """
    else:
        task_text = f"""
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
        """

    image_data_uri = _encode_image(image_path)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_text},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ],
        max_tokens=256
    )

    return response.choices[0].message.content