import json
import numpy as np
import torch
import nltk
import evaluate
from src.original.model import T5ForMultimodalGeneration
from transformers import (
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    T5ForConditionalGeneration
)
from torch.utils.data import Dataset

nltk.download('punkt')

problems = json.load(open("data/scienceqa/data/problems.json"))
name_maps = json.load(open("data/name_map.json"))
captions = json.load(open("data/instruct_captions.json"))["captions"]
image_features = np.load("vision_features/clip.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_one_example(question, context, choice, solution, test_example=True, WithOutput=False, curr_le_data=None):
    input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    output = f"Solution: {solution}"
    
    text = input + f'Solution:'
    text = text.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    return text, output

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics_rougel(eval_preds):
    preds, targets = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    decoded_preds, decoded_labels = postprocess_text(preds, targets)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

class Data(Dataset):
    def __init__(self, target_texts, source_texts, image_ids):
        self.target_texts = target_texts
        self.source_texts = source_texts
        self.image_ids = image_ids
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        target_text = str(self.target_texts[index])
        source_text = str(self.source_texts[index])

        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = tokenizer.batch_encode_plus([source_text],
            max_length=512,
            pad_to_max_length=True,
            truncation=False,
            padding="max_length",
            return_tensors="pt",)

        target = tokenizer.batch_encode_plus([target_text],
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

problems_s = dict()
processed_count = 0
skipped_count = 0
original_order = []

for qid in problems:
    if processed_count < 5 and problems[qid]["split"] == "val":
        if problems[qid]["solution"]:
            problems_s[qid] = problems[qid]
            original_order.append(qid)
            problems_s[qid]['caption'] = captions[qid] if qid in captions else ""
            problems_s[qid]['image_feature'] = image_features[int(name_maps[qid])] str(qid) in name_maps else np.zeros((49, 2048))
            processed_count += 1
        else:
            skipped_count += 1

options = ["A", "B", "C", "D", "E"]
target_texts = []
source_texts = []
image_ids = []

for k in problems_s:
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
    
    prompt, target = create_one_example(question, context, choice_txt, solution)
    
    target_texts.append(target)
    source_texts.append(prompt)
    image_ids.append(image)

tokenizer = AutoTokenizer.from_pretrained("models/mm-cot-large-rationale/mm-cot-large-rationale/")
datacollator = DataCollatorForSeq2Seq(tokenizer)
metric = evaluate.load("rouge")
model = T5ForMultimodalGeneration.from_pretrained(
    "models/mm-cot-large-rationale/mm-cot-large-rationale/",
    patch_size=(49, 2048),
    ignore_mismatched_sizes=True
).to(device)

data = Data(target_texts, source_texts, image_ids)

training_args = Seq2SeqTrainingArguments(
    "Saving",
    do_train=False,
    do_eval=False,
    evaluation_strategy="no",
    logging_strategy="steps",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    eval_accumulation_steps=None,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=50,
    predict_with_generate=True,
    generation_max_length=256,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=datacollator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_rougel
)

predict_results = trainer.predict(test_dataset=data)

preds, targets = predict_results.predictions, predict_results.label_ids
preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
preds = tokenizer.batch_decode(
    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
targets = tokenizer.batch_decode(
    targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
preds = [pred.strip() for pred in preds]

output_data = {}
for i, qid in enumerate(original_order):
    output_data[qid] = {
        "generated_rationale": preds[i],
        "gold_label_rationale": targets[i]
    }

with open("Saving/predictions_ans_eval.json", "w") as writer:
    writer.write(json.dumps(output_data, indent=4))

print(f"{len(preds)} rationales generated and {skipped_count} rationales skipped due to missing solution in data source.")