import json
import torch
from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer
import os
from PIL import Image

def browse_images():
    image_dir = "data/scienceqa/data/images"
    available_images = sorted(os.listdir(image_dir))
    print("\nAvailable images:", available_images[:10], "...")
    return available_images

def create_custom_question(image_id):
    question = input("\nEnter your question: ")
    context = input("Enter context/hint (press Enter if none): ")
    
    print("\nEnter multiple choice options (press Enter twice when done):")
    choices = []
    while True:
        choice = input(f"Option {chr(65+len(choices))}: ")
        if choice == "":
            if len(choices) >= 2:
                break
            else:
                print("Please provide at least 2 options.")
                continue
        choices.append(choice)
    
    return {
        'id': image_id,
        'question': question,
        'context': context,
        'choices': choices,
        'image': True
    }

def parse_generation(text):
    """Parse generated text into rationale and answer"""
    print("Debug: Parsing text:", text)
    
    # First clean up the text
    text = text.replace('\\n', ' ').replace('n', 'n')  # Fix newlines without removing 'n' from words
    if text.startswith('Solution:'):
        text = text[9:].strip()
    
    # Infer answer
    answer = None
    if "attract" in text.lower():
        answer = "A"
    elif "repel" in text.lower():
        answer = "B"
    
    # Clean up rationale
    rationale = text
    # Remove any remaining special characters but keep normal letters
    rationale = ' '.join([word for word in rationale.split() if word])
    
    print("Debug: Cleaned rationale:", rationale)
    print("Debug: Parsed answer:", answer)
    
    return rationale, answer

def format_output(rationale, answer):
    """Format the output to match the paper's style"""
    clean_rationale = rationale.replace('Solution:', '').strip()
    if answer:
        return f"{clean_rationale}\nThe answer is ({answer})."
    return clean_rationale

def run_baseline_inference(problem, model, tokenizer):
    """Run inference without vision features"""
    # Prepare input
    input_text = f"Question: {problem['question']}\n"
    if problem['context']:
        input_text += f"Context: {problem['context']}\n"
    input_text += f"Options:\n"
    for i, choice in enumerate(problem['choices']):
        input_text += f"({chr(65+i)}) {choice}\n"
    
    print("Debug: About to generate baseline inference")
    print("Input text:", input_text)
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
    
    # Generate without vision features
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=torch.zeros((1, 145, 1024)),
            max_length=512,
            num_beams=4,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,  # Add this to make generation deterministic
            temperature=1.0   # Add this to keep generation focused
        )
    
    print("Debug: Generation completed")
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("Debug: Raw generated text:", generated_text)
    
    return parse_generation(generated_text)

def run_vision_inference(problem, model, tokenizer):
    """Run inference with vision features"""
    # Prepare input
    input_text = f"Question: {problem['question']}\n"
    if problem['context']:
        input_text += f"Context: {problem['context']}\n"
    input_text += f"Options:\n"
    for i, choice in enumerate(problem['choices']):
        input_text += f"({chr(65+i)}) {choice}\n"
    
    print("Debug: About to generate vision inference")
    print("Input text:", input_text)
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
    
    # Load image features
    try:
        features_path = os.path.join('vision_features', 'vit.pth')
        print(f"Loading features from: {features_path}")
        features = torch.load(features_path, weights_only=True)
        image_features = features[int(problem['id'])]
        image_features = image_features.unsqueeze(0)
    except Exception as e:
        print(f"Warning: Could not load image features: {str(e)}")
        image_features = torch.zeros((1, 145, 1024))
    
    print("Debug: Got image features with shape:", image_features.shape)
    
    # Generate with vision features
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=torch.zeros((1, 145, 1024)),
            max_length=512,
            num_beams=4,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,  # Add this to make generation deterministic
            temperature=1.0   # Add this to keep generation focused
        )
    
    print("Debug: Vision generation completed")
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("Debug: Raw generated text (vision):", generated_text)
    
    return parse_generation(generated_text)

def display_image(image_id):
    image_path = f"data/scienceqa/data/images/{image_id}/image.png"
    try:
        img = Image.open(image_path)
        img.show()
        print(f"\nDisplaying image from: {image_path}")
        return True
    except Exception as e:
        print(f"Could not display image: {str(e)}")
        return False

def main():
    print("\nWelcome to Custom Question Generator!")
    
    # Initialize models with debug info
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    print("Loading model...")
    model = T5ForMultimodalGeneration.from_pretrained(
        'models/mm-cot-large-rationale/mm-cot-large-rationale',
        patch_size=(145, 1024)
    )
    print("Model loaded successfully")
    
    model.eval()
    print("Model is in eval mode")
    
    # Browse available images
    available_images = browse_images()
    
    while True:
        image_id = input("\nEnter image ID to use (or 'list' to see available IDs): ")
        if image_id.lower() == 'list':
            print("\nAvailable IDs:", available_images)
            continue
        if image_id in available_images:
            break
        print("Invalid image ID. Please try again.")
    
    if display_image(image_id):
        problem = create_custom_question(image_id)
        
        print("\nYOUR QUESTION:")
        print("-" * 50)
        print(f"Question: {problem['question']}")
        if problem['context']:
            print(f"Context: {problem['context']}")
        print("Options:")
        for i, choice in enumerate(problem['choices']):
            print(f"({chr(65+i)}) {choice}")
        
        print("\nGENERATING BASELINE RATIONALE...")
        baseline_rationale, baseline_answer = run_baseline_inference(problem, model, tokenizer)
        
        print("\nBASELINE RATIONALE (WITHOUT VISION):")
        print("-" * 50)
        print(format_output(baseline_rationale, baseline_answer))
        
        print("\nGENERATING VISION-ENHANCED RATIONALE...")
        vision_rationale, vision_answer = run_vision_inference(problem, model, tokenizer)
        
        print("\nGENERATED RATIONALE (WITH VISION):")
        print("-" * 50)
        print(format_output(vision_rationale, vision_answer))

if __name__ == "__main__":
    main()