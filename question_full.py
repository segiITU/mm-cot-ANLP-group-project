import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import timm  # Use installed version
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as T
from PIL import Image
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from src.original.model import T5ForMultimodalGeneration

QUESTION = "Will these magnets attract or repel each other?"
CONTEXT = "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart."
OPTIONS = ["attract", "repel"]
IMAGE_ID = "173"

def extract_features(image_path):
    """Extract features using ViT"""
    # Initialize model
    vit_model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0)
    vit_model.eval()
    
    # Setup transform
    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)
    
    # Load and transform image
    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        features = vit_model.forward_features(input_tensor)
    
    return features

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    # Rationale generation model (multimodal)
    rationale_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-rationale/mm-cot-large-rationale'),
        patch_size=(145, 1024)
    )
    rationale_model.eval()
    
    # Answer inference model (language-only)
    answer_model = T5ForConditionalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-answer/mm-cot-large-answer')
    )
    answer_model.eval()
    
    return tokenizer, rationale_model, answer_model

def load_vision_features(image_id):
    # Get image path
    image_path = os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'image.png')
    if not os.path.exists(image_path):
        image_path = os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'choice_0.png')
    
    if not os.path.exists(image_path):
        print(f"Error: No image found for ID {image_id}")
        return torch.zeros((1, 145, 1024))
    
    print(f"Loading image from: {image_path}")
    
    try:
        features = extract_features(image_path)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return torch.zeros((1, 145, 1024))

def generate_rationale(tokenizer, model, question, context, options, vision_features):
    input_text = f"Question: {question}\nContext: {context}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=vision_features,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            use_cache=True
        )
    
    rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rationale

def generate_answer(tokenizer, model, question, context, options, rationale):
    input_text = f"Question: {question}\nContext: {context}"
    input_text += f"\nRationale: {rationale}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            use_cache=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def format_output(rationale, answer):
    return f"Rationale: {rationale}\nAnswer: The answer is ({answer})."

def main():
    print("\nLoading models...\n")
    tokenizer, rationale_model, answer_model = load_models()
    
    print(f"\nExtracting vision features for image {IMAGE_ID}...\n")
    vision_features = load_vision_features(IMAGE_ID)
    
    print("\nStage 1: Generating rationale...\n")
    rationale = generate_rationale(
        tokenizer, 
        rationale_model,
        QUESTION, 
        CONTEXT, 
        OPTIONS, 
        vision_features
    )
    print(f"\nGenerated rationale:\n{rationale}")
    
    print("\nStage 2: Generating answer...\n")
    answer = generate_answer(
        tokenizer,
        answer_model,
        QUESTION,
        CONTEXT,
        OPTIONS,
        rationale
    )
    print(f"\nGenerated answer: {answer}\n")
    
    print("\nFinal output:")
    print("-" * 50)
    print(format_output(rationale, answer))

if __name__ == "__main__":
    main()