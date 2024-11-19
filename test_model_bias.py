import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from original.model import T5ForMultimodalGeneration

TEST_CASES = [
    {
        "question": "Will these magnets attract or repel each other?",
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["attract", "repel"],
        "image_id": "86"
    },
    {
        "question": "Will these magnets attract or repel each other?",
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["repel", "attract"],  # Swapped options
        "image_id": "86"
    },
    {
        "question": "Will these magnets repel or attract each other?",  # Swapped word order
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["repel", "attract"],
        "image_id": "86"
    },
    {
        "question": "How are these magnets interacting?",  # Different phrasing
        "context": "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart.",
        "options": ["attract", "repel"],
        "image_id": "86"
    }
]

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    # Both models need to be T5ForMultimodalGeneration since it expects image_ids
    rationale_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-rationale/mm-cot-large-rationale'),
        patch_size=(145, 1024)
    )
    rationale_model.eval()
    
    answer_model = T5ForMultimodalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-answer/mm-cot-large-answer'),
        patch_size=(145, 1024)
    )
    answer_model.eval()
    
    return tokenizer, rationale_model, answer_model

def load_vision_features(image_id):
    features_path = os.path.join(project_root, 'vision_features', 'vit.pth')
    try:
        features = torch.load(features_path)
        vision_features = features[int(image_id)]
        return vision_features.unsqueeze(0)
    except Exception as e:
        print(f"Error loading vision features: {e}")
        return torch.zeros((1, 145, 1024))

def generate_rationale(tokenizer, model, question, context, options, vision_features=None):
    input_text = f"Question: {question}\nContext: {context}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Always provide image_ids, but use zeros for no-vision case
    if vision_features is None:
        vision_features = torch.zeros((1, 145, 1024))
    
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
    
    # Always provide dummy image_ids for answer model
    dummy_vision = torch.zeros((1, 145, 1024))
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=dummy_vision,
            max_length=64,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            use_cache=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def format_output(rationale, answer, test_case):
    option_mapping = {f"({chr(65+i)})": opt for i, opt in enumerate(test_case["options"])}
    formatted_answer = answer
    for key, value in option_mapping.items():
        if key in answer:
            formatted_answer = value
    return f"Question: {test_case['question']}\nOptions: {test_case['options']}\nRationale: {rationale}\nPredicted: {formatted_answer}"

def run_test_case(tokenizer, rationale_model, answer_model, test_case, use_vision=True):
    if use_vision:
        vision_features = load_vision_features(test_case["image_id"])
    else:
        vision_features = None
        
    rationale = generate_rationale(
        tokenizer, 
        rationale_model,
        test_case["question"], 
        test_case["context"], 
        test_case["options"],
        vision_features
    )
    
    answer = generate_answer(
        tokenizer,
        answer_model,
        test_case["question"],
        test_case["context"],
        test_case["options"],
        rationale
    )
    
    return format_output(rationale, answer, test_case)

def main():
    print("\nLoading models...\n")
    tokenizer, rationale_model, answer_model = load_models()
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"\nTest Case {i+1}")
        print("=" * 50)
        
        print("\nWithout Vision Features:")
        print("-" * 30)
        result_no_vision = run_test_case(
            tokenizer, 
            rationale_model, 
            answer_model, 
            test_case, 
            use_vision=False
        )
        print(result_no_vision)
        
        print("\nWith Vision Features:")
        print("-" * 30)
        result_with_vision = run_test_case(
            tokenizer, 
            rationale_model, 
            answer_model, 
            test_case, 
            use_vision=True
        )
        print(result_with_vision)
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()