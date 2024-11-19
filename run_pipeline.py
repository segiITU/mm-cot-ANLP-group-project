import os
import sys
import torch
from transformers import AutoTokenizer
from PIL import Image
import timm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'original'))
sys.path.append(os.path.join(project_root, 'models'))

from model import T5ForMultimodalGeneration

def display_image(image_path):
   try:
       img = Image.open(image_path)
       img.show()
       print(f"Displaying image from: {image_path}")
   except Exception as e:
       print(f"Error displaying image: {e}")

def extract_image_features(image_path, vit_features_path="vision_features/vit.pth"):
   try:
       all_features = torch.load(vit_features_path)
       image_id = int(image_path.split('/')[-2])
       image_features = all_features[image_id]
       print(f"\nLoaded pre-extracted features shape: {image_features.shape}")
       return image_features.unsqueeze(0)
   except Exception as e:
       print(f"\nError loading features: {str(e)}")
       return None

def run_mm_cot(image_path, question, context, options):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("Loading tokenizer...")
   tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-large")
   
   print("Loading custom model...")
   rationale_model = T5ForMultimodalGeneration.from_pretrained(
       "models/mm-cot-base-rationale",
       patch_size=(145, 1024),
       local_files_only=True
   ).to(device)
   
   answer_model = T5ForMultimodalGeneration.from_pretrained(
       "models/mm-cot-base-ans",
       patch_size=(145, 1024),
       local_files_only=True
   ).to(device)

   input_text = (
       f"Question: {question}\n"
       f"Context: {context}\n"
       f"Options: {', '.join(f'({chr(65+i)}) {opt}' for i, opt in enumerate(options))}\n"
       f"Let's analyze this step by step:"
   )
   
   vision_features = extract_image_features(image_path)
   if vision_features is not None:
       vision_features = vision_features.to(device)
   else:
       vision_features = torch.zeros((1, 145, 1024)).to(device)
       
   inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   
   with torch.no_grad():
       outputs = rationale_model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           image_ids=vision_features,
           max_length=128,
           do_sample=False
       )
   
   rationale = tokenizer.decode(outputs[0], skip_special_tokens=True)

   input_text_with_rationale = (
       f"Question: {question}\n"
       f"Context: {context}\n"
       f"Rationale: {rationale}\n"
       f"Options: {', '.join(f'({chr(65+i)}) {opt}' for i, opt in enumerate(options))}\n"
       f"Based on this analysis, the answer is:"
   )

   inputs = tokenizer(input_text_with_rationale, return_tensors="pt", max_length=512, truncation=True)
   inputs = {k: v.to(device) for k, v in inputs.items()}

   with torch.no_grad():
       outputs = answer_model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           image_ids=vision_features,
           max_length=32,
           do_sample=False
       )

   answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
   return {'rationale': rationale, 'answer': answer}

if __name__ == "__main__":
   image_path = "data/scienceqa/data/images/208/image.png"
   question = "Will these magnets attract or repel each other?"
   context = "Two magnets are placed as shown."
   options = ["attract", "repel"]
   
   # Display image first
   display_image(image_path)
   
   print("\nQuestion:", question)
   print("Context:", context) 
   print("Options:", ', '.join(f'({chr(65+i)}) {opt}' for i, opt in enumerate(options)))
   
   result = run_mm_cot(image_path, question, context, options)
   print(f"\nRationale: {result['rationale']}")
   print(f"\nAnswer: {result['answer']}")