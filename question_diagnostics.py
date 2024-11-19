import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as T
from PIL import Image
import os
import sys
import json
from pprint import pprint

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'original'))
from src.original.model import T5ForMultimodalGeneration

def load_caption(image_id):
    """Load InstructBLIP caption for the image"""
    caption_path = os.path.join(project_root, 'data', 'instruct_captions.json')
    try:
        with open(caption_path, 'r') as f:
            captions = json.load(f)
            return captions.get(str(image_id), "")
    except Exception as e:
        print(f"Warning: Could not load caption: {e}")
        return ""

TEST_CASES = [
    {
        "name": "Base Case - No Vision",
        "question": "Will these magnets attract or repel each other?",
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["attract", "repel"],
        "image_id": None,
        "description": "Testing model behavior with no image features and no caption"
    },
    {
        "name": "Regular Image + Caption",
        "question": "Will these magnets attract or repel each other?",
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["attract", "repel"],
        "image_id": "86",
        "description": "Testing with standard magnet image and its caption"
    },
    {
        "name": "Corrupted Features + Caption",
        "question": "Will these magnets attract or repel each other?",
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["attract", "repel"],
        "image_id": "86",
        "corrupt_features": True,
        "description": "Testing with corrupted vision features but correct caption"
    },
    {
        "name": "Image Only (No Context)",
        "question": "Will these magnets attract or repel each other?",
        "context": "",
        "options": ["attract", "repel"],
        "image_id": "86",
        "description": "Testing with only image/caption, no additional context"
    }
]

def extract_features(image_path):
    """Extract features using ViT with detailed diagnostics"""
    print(f"\nExtracting features from: {image_path}")
    
    vit_model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0)
    vit_model.eval()
    
    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)
    
    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        features = vit_model.forward_features(input_tensor)
        
        print(f"Feature shape: {features.shape}")
        print(f"Feature stats:")
        print(f"  Min: {features.min().item():.3f}")
        print(f"  Max: {features.max().item():.3f}")
        print(f"  Mean: {features.mean().item():.3f}")
        print(f"  Std: {features.std().item():.3f}")
    
    return features

def load_vision_features(test_case):
    """Load vision features with comprehensive error handling and diagnostics"""
    image_id = test_case.get('image_id')
    
    if image_id is None:
        print("\nNo image ID provided, using zero tensor")
        return torch.zeros((1, 145, 1024))
    
    image_path = os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'image.png')
    if not os.path.exists(image_path):
        image_path = os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'choice_0.png')
    
    if not os.path.exists(image_path):
        print(f"\nError: No image found for ID {image_id}")
        print(f"Tried paths:")
        print(f"  - {os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'image.png')}")
        print(f"  - {os.path.join(project_root, 'data', 'scienceqa', 'data', 'images', str(image_id), 'choice_0.png')}")
        return torch.full((1, 145, 1024), -999.0)
    
    try:
        features = extract_features(image_path)
        if test_case.get('corrupt_features'):
            features = features * -1
            print("\nFeatures deliberately corrupted for testing")
        return features
    except Exception as e:
        print(f"\nError extracting features: {str(e)}")
        return torch.full((1, 145, 1024), -999.0)

def format_input_QCM_E(question, context, options, caption=""):
    """Format following QCM-E prompt format for rationale generation"""
    input_text = f"Question: {question}\n"
    if context:
        input_text += f"Context: {context}\n"
    if caption:
        input_text += f"Image: {caption}\n"
    input_text += "Options:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    input_text += "\nPlease explain step by step:"
    return input_text

def format_input_QCMG_A(question, context, options, rationale, caption=""):
    """Format following QCMG-A prompt format for answer generation"""
    input_text = f"Question: {question}\n"
    if context:
        input_text += f"Context: {context}\n"
    if caption:
        input_text += f"Image: {caption}\n"
    input_text += "Options:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    input_text += f"\nRationale: {rationale}\nTherefore, the answer is:"
    return input_text

def generate_rationale(tokenizer, model, test_case, vision_features):
    """Generate rationale using QCM-E format"""
    caption = load_caption(test_case.get('image_id')) if test_case.get('image_id') else ""
    
    input_text = format_input_QCM_E(
        test_case["question"],
        test_case["context"],
        test_case["options"],
        caption
    )
    
    print("\nRationale Generation Input:")
    print("-" * 50)
    print(input_text)
    print("-" * 50)
    
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

def generate_answer(tokenizer, model, test_case, rationale):
    """Generate answer using QCMG-A format"""
    caption = load_caption(test_case.get('image_id')) if test_case.get('image_id') else ""
    
    input_text = format_input_QCMG_A(
        test_case["question"],
        test_case["context"],
        test_case["options"],
        rationale,
        caption
    )
    
    print("\nAnswer Generation Input:")
    print("-" * 50)
    print(input_text)
    print("-" * 50)
    
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

def load_models():
    """Load models with diagnostic information"""
    print("\nLoading models...")
    
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    print("\nLoading rationale model...")
    rationale_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-rationale/mm-cot-large-rationale'),
        patch_size=(145, 1024)
    )
    rationale_model.eval()
    
    print("\nLoading answer model...")
    answer_model = T5ForConditionalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-answer/mm-cot-large-answer')
    )
    answer_model.eval()
    
    return tokenizer, rationale_model, answer_model

def run_diagnostics():
    """Run comprehensive diagnostics with proper prompt formats"""
    results = []
    
    tokenizer, rationale_model, answer_model = load_models()
    
    for test_case in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"Running test case: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*80}")
        
        vision_features = load_vision_features(test_case)
        
        print("\nGenerating rationale...")
        rationale = generate_rationale(
            tokenizer,
            rationale_model,
            test_case,
            vision_features
        )
        
        print("\nGenerating answer...")
        answer = generate_answer(
            tokenizer,
            answer_model,
            test_case,
            rationale
        )
        
        result = {
            "test_case": test_case["name"],
            "description": test_case["description"],
            "image_id": test_case["image_id"],
            "question": test_case["question"],
            "options": test_case["options"],
            "caption": load_caption(test_case.get('image_id')) if test_case.get('image_id') else "",
            "rationale": rationale,
            "answer": answer,
            "vision_features_shape": list(vision_features.shape),
            "vision_features_stats": {
                "min": float(vision_features.min()),
                "max": float(vision_features.max()),
                "mean": float(vision_features.mean()),
                "std": float(vision_features.std())
            }
        }
        
        results.append(result)
        
        print("\nTest case result:")
        print("-" * 50)
        pprint(result)
    
    # Save results to file
    output_file = "diagnostics_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results

if __name__ == "__main__":
    results = run_diagnostics()