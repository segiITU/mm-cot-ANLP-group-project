import json
import torch
from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer
import os

def load_example(problems_file="data/scienceqa/data/problems.json"):
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    # Find a problem with an image
    for pid, problem in problems.items():
        if problem.get('image'):
            # Found a problem with an image
            return {
                'id': pid,  # Store the problem ID
                'question': problem['question'],
                'context': problem.get('hint', 'N/A'),
                'choices': problem['choices'],
                'answer': problem['answer'],
                'image': True  # Just store boolean since we'll use pid for features
            }

def run_inference(problem):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    model = T5ForMultimodalGeneration.from_pretrained(
        'models/mm-cot-large-rationale/mm-cot-large-rationale',
        patch_size=(145, 1024)
    )
    model.eval()
    
    # Prepare input
    input_text = f"Question: {problem['question']}\n"
    input_text += f"Context: {problem['context']}\n"
    input_text += f"Options:\n"
    for i, choice in enumerate(problem['choices']):
        input_text += f"({chr(65+i)}) {choice}\n"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
    
    # Load image features if present
    if problem['image']:
        try:
            features_path = os.path.join('vision_features', 'vit.pth')
            print(f"Loading features from: {features_path}")
            features = torch.load(features_path, weights_only=True)  # Add weights_only=True
            image_features = features[int(problem['id'])]  # Use problem ID to index features
            image_features = image_features.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Warning: Could not load image features: {str(e)}")
            image_features = torch.zeros((1, 145, 1024))  # Create dummy tensor
    else:
        image_features = torch.zeros((1, 145, 1024))
    
    # Generate rationale
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=image_features,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rationale

def main():
    # Load and run example
    problem = load_example()
    print("\nINPUT:")
    print("-" * 50)
    print(f"Question: {problem['question']}")
    print(f"Context: {problem['context']}")
    print("Options:")
    for i, choice in enumerate(problem['choices']):
        print(f"({chr(65+i)}) {choice}")
    if problem['image']:
        print(f"[Image is present for problem ID: {problem['id']}]")
    
    print("\nGENERATING RATIONALE...")
    rationale = run_inference(problem)
    
    print("\nOUTPUT:")
    print("-" * 50)
    print(rationale)
    print(f"\nCorrect Answer: {chr(65 + problem['answer'])}")

if __name__ == "__main__":
    main()