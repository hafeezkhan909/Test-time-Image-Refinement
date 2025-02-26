import torch

def encode_prompt_sdxl(prompt, tokenizers, text_encoders, device):
    text_inputs = []
    for tokenizer in tokenizers:
        inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_inputs.append(inputs)
    
    text_embeddings = []
    for inputs, text_encoder in zip(text_inputs, text_encoders):
        text_embeddings.append(text_encoder(inputs.input_ids.to(device))[0])
    
    return torch.cat(text_embeddings, dim=-1)