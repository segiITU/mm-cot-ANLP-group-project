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
import datamaker

nltk.download('punkt')

problems = json.load(open("data/scienceqa/data/problems.json"))
name_maps = json.load(open("data/name_map.json"))
captions = json.load(open("data/instruct_captions.json"))["captions"]
image_features = np.load("vision_features/clip.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("models/mm-cot-large-rationale/mm-cot-large-rationale/")
datacollator = DataCollatorForSeq2Seq(tokenizer)

data,_ = datamaker.datamaker(problems, name_maps, captions, image_features, tokenizer)

metric = evaluate.load("rouge")

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

model = T5ForMultimodalGeneration.from_pretrained(
    "models/mm-cot-large-rationale/mm-cot-large-rationale/",
    patch_size=(49, 2048),
    ignore_mismatched_sizes=True
).to(device)

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