from typing import List, Tuple, Union
from PIL import Image
import base64
import io
import requests

def process_vision_info(messages) -> Tuple[List[Union[Image.Image, None]], List[Union[str, None]]]:
    """
    Process vision information from messages.
    
    Args:
        messages: List of message objects containing text and image content
        
    Returns:
        Tuple containing:
        - List of PIL Image objects or None for non-image messages
        - List of video paths or None for non-video messages
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        # Handle string content case
        if isinstance(message['content'], str):
            continue
            
        for content in message['content']:
            if content.get("type") == "image_url" and content.get("image_url") and content["image_url"].get("url"):
                # Handle image URLs
                try:
                    response = requests.get(content["image_url"]["url"])
                    image = Image.open(io.BytesIO(response.content))
                    image_inputs.append(image)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
            elif content.get("type") == "image" and content.get("image"):
                # Handle base64 encoded images
                try:
                    if content["image"].startswith('data:image'):
                        base64_data = content["image"].split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        image_inputs.append(image)
                    elif content["image"].startswith(('http://', 'https://')):
                        response = requests.get(content["image"])
                        image = Image.open(io.BytesIO(response.content))
                        image_inputs.append(image)
                    elif content["image"].startswith('file://'):
                        image_path = content["image"].replace('file://', '')
                        image = Image.open(image_path)
                        image_inputs.append(image)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
    
    # Only append video inputs if we actually have images
    video_inputs = [None] * len(image_inputs) if image_inputs else []
    
    return image_inputs, video_inputs