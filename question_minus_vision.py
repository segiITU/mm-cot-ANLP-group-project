import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from original.model import T5ForMultimodalGeneration

QUESTION = "Will these magnets attract or repel each other?"
CONTEXT = "Two magnets are placed as shown. Hint: Magnets that attract pull together. Magnets that repel push apart."
OPTIONS = ["repel", "attract"]

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
    
    # Rationale generation model (baseline without vision)
    rationale_model = T5ForConditionalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-rationale/mm-cot-large-rationale')
    )
    rationale_model.eval()
    
    # Answer inference model (language-only)
    answer_model = T5ForConditionalGeneration.from_pretrained(
        os.path.join(project_root, 'models/mm-cot-large-answer/mm-cot-large-answer')
    )
    answer_model.eval()
    
    return tokenizer, rationale_model, answer_model

def generate_rationale(tokenizer, model, question, context, options):
    input_text = f"Question: {question}\nContext: {context}\nOptions:"
    for i, opt in enumerate(options):
        input_text += f"\n({chr(65+i)}) {opt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
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
    print("\nLoading models (baseline without vision)...\n")
    tokenizer, rationale_model, answer_model = load_models()
    
    print("\nStage 1: Generating rationale...\n")
    rationale = generate_rationale(
        tokenizer, 
        rationale_model,
        QUESTION, 
        CONTEXT, 
        OPTIONS
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