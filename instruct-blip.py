import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
import json

def generate_captions(image_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Add num_query_tokens attribute to the processor
    processor.num_query_tokens = model.config.num_query_tokens
    
    # Expand model embeddings layer to add special <image> token
    model.config.image_token_index = model.config.vocab_size
    model.resize_token_embeddings(model.config.vocab_size + 1)
    
    model.to(device)
    
    captions = {}
    for image_id in image_ids:
        image_path = f"data/scienceqa/data/images/{image_id}/image.png"
        
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            prompt = "Write a detailed description."
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            
            print(f"Inputs for image {image_id}: {inputs}")
            
            outputs = model.generate(**inputs, max_new_tokens=100)
            print(f"Outputs for image {image_id}: {outputs}")
            
            caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"Decoded caption for image {image_id}: {caption}")
            
            captions[str(image_id)] = caption
            print(f"\nImage {image_id} caption: {caption}")

    with open('data/instruct_captions2.json', 'w') as f:
        json.dump(captions, f, indent=2)

if __name__ == "__main__":
    image_ids = [86] #, 173, 200, 208
    generate_captions(image_ids)