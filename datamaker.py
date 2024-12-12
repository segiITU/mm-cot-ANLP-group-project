import torch
import nltk
from torch.utils.data import Dataset
import numpy as np

nltk.download('punkt')

def create_one_example(question, context, choice, solution, answer = None, curr_le_data=None):
    #Check if we are doing Answers or Rationale
    if curr_le_data:
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        output = f"Answer: The answer is {answer}."
        text = input + f'Answer:'
    else:
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
        output = f"Solution: {solution}"
        text = input + f'Solution:'
    text = text.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    return text, output


class Data(Dataset):
    def __init__(self, target_texts, source_texts, image_ids, tokenizer):
        self.target_texts = target_texts
        self.source_texts = source_texts
        self.image_ids = image_ids
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        target_text = str(self.target_texts[index])
        source_text = str(self.source_texts[index])

        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text],
            max_length=512,
            pad_to_max_length=True,
            truncation=False,
            padding="max_length",
            return_tensors="pt",)

        target = self.tokenizer.batch_encode_plus([target_text],
            max_length=256,  
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": torch.tensor(image_id).squeeze(),
            "labels": target_ids,
        }
    
    def __len__(self):
        return len(self.target_texts)

def datamaker(problems, name_maps, captions, image_features, tokenizer, ans=False, rationale=None):
    problems_s = dict()
    processed_count = 0
    skipped_count = 0
    original_order = []
    
    for qid in problems:
        if processed_count < 10 and problems[qid]["split"] == "val" and problems[qid]["image"]:
            if problems[qid]["solution"]:
                problems_s[qid] = problems[qid]
                original_order.append(qid)
                problems_s[qid]['caption'] = captions[qid] if qid in captions else ""
                problems_s[qid]['image_feature'] = image_features[int(name_maps[qid])] if str(qid) in name_maps else np.zeros((49, 2048))
                processed_count += 1
            else:
                skipped_count += 1
    options = ["A", "B", "C", "D", "E"]
    target_texts = []
    source_texts = []
    image_ids = []
    key_pos = dict()

    for i,k in enumerate(problems_s):
        question = problems_s[k]["question"]
        txt_context = problems_s[k]['hint']
        img_context = problems_s[k]['caption']
        context = " ".join([txt_context, img_context]).strip()
        if context == "":
            context = "N/A"
    
        choices = problems_s[k]['choices']
        choice_list = []
        for i, c in enumerate(problems_s[k]['choices']):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
    
        solution = problems_s[k]["solution"]
        image = problems_s[k]["image_feature"]
        
        answer = "(" + options[problems_s[k]['answer']] + ")"
        if ans:
            prompt,target=create_one_example(question, context, choice_txt, solution, answer, rationale[k]["generated_rationale"])
        else:
            prompt, target = create_one_example(question, context, choice_txt, solution)
        
        target_texts.append(target)
        source_texts.append(prompt)
        image_ids.append(image)
        key_pos[k]=i

    data = Data(target_texts, source_texts, image_ids, tokenizer)
    return data, key_pos