import os
import sys
import torch
from transformers import AutoTokenizer
from PIL import Image
import timm

# Adjust the paths as necessary
project_root = os.path.dirname(os.path.abspath(__file__))

# Add paths to sys.path
sys.path.append(os.path.join(project_root, 'src', 'original'))
sys.path.append(os.path.join(project_root, 'models'))

from extract_features import extract_features
from model import T5ForMultimodalGeneration  # Update this import based on your project structure

# Constants
QUESTION = "Will these magnets attract or repel each other?"
CONTEXT = "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart."
OPTIONS = ["attract", "repel"]
IMAGE_ID = "86"  # Updated to match the correct image ID

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    # Updated patch_size to match the new embedding dimension
    rationale_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models', 'mm-cot-large-rationale', 'mm-cot-large-rationale'),
        patch_size=(197, 1024)
    )
    rationale_model.eval()
    
    answer_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models', 'mm-cot-large-answer', 'mm-cot-large-answer'),
        patch_size=(197, 1024)
    )
    answer_model.eval()
    
    return tokenizer, rationale_model, answer_model

def load_vision_features(image_id):
    # Get image path
    image_path = os.path.join(
        project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'image.png'
    )
    if not os.path.exists(image_path):
        image_path = os.path.join(
            project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'choice_0.png'
        )

    if not os.path.exists(image_path):
        print(f"Error: No image found for ID {image_id}")
        return None

    print(f"Loading image from: {image_path}")

    try:
        features = extract_features('vit', image_path)
        print("Extracted features shape:", features.shape)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_features(img_type, image_path):
    if img_type == 'vit':
        import timm
        import torch
        from PIL import Image
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        # Load the pre-trained ViT model with embed_dim=1024
        vit_model = timm.create_model('vit_large_patch16_224', pretrained=True)
        vit_model.eval()

        # Preprocess the image
        config = resolve_data_config({}, model=vit_model)
        transform = create_transform(**config)
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            features = vit_model.forward_features(img)

        return features
    else:
        raise NotImplementedError(f"Image type '{img_type}' is not supported.")

def generate_rationale(tokenizer, model, question, context, options, vision_features):
    input_text = f"Question: {question}\nContext: {context}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    input_text += "\nLet's think step by step."

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_ids=vision_features,
            max_length=256,
            num_beams=4,
            temperature=0.7
        )

    rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rationale.strip()

def generate_answer(tokenizer, model, question, context, options, rationale, vision_features):
    input_text = f"Question: {question}\nContext: {context}"
    input_text += f"\nRationale: {rationale}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    input_text += "\nTherefore, the answer is"

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_ids=vision_features,
            max_length=64,
            num_beams=4,
            temperature=0.7
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def format_output(rationale, answer):
    return f"Rationale:\n{rationale}\n\nAnswer: The answer is {answer}."

def main():
    print("\nLoading models...\n")
    tokenizer, rationale_model, answer_model = load_models()
    
    print(f"\nExtracting vision features for image {IMAGE_ID}...\n")
    vision_features = load_vision_features(IMAGE_ID)
    if vision_features is None:
        print("Using zero tensor for vision features.")
        # Adjust dimensions based on expected input
        vision_features = torch.zeros((1, 197, 1024))  # Update dimensions if necessary
    
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
        rationale,
        vision_features
    )
    print(f"\nGenerated answer: {answer}\n")
    
    print("\nFinal output:")
    print("-" * 50)
    print(format_output(rationale, answer))

if __name__ == "__main__":
    main()