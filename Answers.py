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
import re
import pandas as pd
import datamaker

nltk.download('punkt')

problems = json.load(open("data/scienceqa/data/problems.json"))
name_maps = json.load(open("data/name_map.json"))
captions = json.load(open("data/instruct_captions.json"))["captions"]
image_features = np.load("vision_features/clip.npy")
rationale = json.load(open("Saving/predictions_ans_eval.json"))
options = ["A", "B", "C", "D", "E"]

def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)
        
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED" 
    return answer

def compute_metrics_acc(eval_preds):
    preds, targets = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    correct = 0
    assert len(preds) == len(targets)
    for idx, pred in enumerate(preds):
        reference = targets[idx]
        reference = extract_ans(reference)
        extract_pred = extract_ans(pred)
        best_option = extract_pred
        if reference == best_option:
            correct +=1 
    return {'accuracy': 1.0*correct/len(targets)}

def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    if len(total_pd) == 0:
        acc=0
    else:
        acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc

def get_scores(result_data, rationale_data, results_reference, data_file):
    # read result file
    results = result_data
    num = len(results)
#    assert num == 4241
    #print("number of questions:", num)

    # read data file
    sqa_data = json.load(open(data_file))
    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data)[list(results.keys())].T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set
    # update data
    for index, row in res_pd.iterrows():
        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
    #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)

    scores = {
            "answer":{
                'acc_natural':
                get_acc_with_contion(res_pd, 'subject', 'natural science'),
                'acc_social':
                get_acc_with_contion(res_pd, 'subject', 'social science'),
                'acc_language':
                get_acc_with_contion(res_pd, 'subject', 'language science'),
                'acc_has_text':
                get_acc_with_contion(res_pd, 'has_text', True),
                'acc_has_image':
                get_acc_with_contion(res_pd, 'has_image', True),
                'acc_no_context':
                get_acc_with_contion(res_pd, 'no_context', True),
                'acc_grade_1_6':
                get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
                'acc_grade_7_12':
                get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
                'acc_average':
                "{:.2f}".format(acc_average),
            }}

    return scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("models/mm-cot-large-answer/mm-cot-large-answer/")
datacollator = DataCollatorForSeq2Seq(tokenizer)
data, key_pos = datamaker.datamaker(problems, name_maps, captions, image_features, tokenizer, ans=True, rationale=rationale)

metric = evaluate.load("rouge")
model = T5ForMultimodalGeneration.from_pretrained(
    "models/mm-cot-large-answer/mm-cot-large-answer/",
    patch_size=(49, 2048),
    ignore_mismatched_sizes=True
).to(device)

#Same as generate, but max length is 64.
training_args = Seq2SeqTrainingArguments(
            "Saving",
            do_train=False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= 5e-5,
            eval_accumulation_steps=None,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            num_train_epochs=50,
            predict_with_generate=True,
            generation_max_length=64,
            report_to="none",
        )

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc
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


results_ans = {}
results_rationale = {}
results_reference = {}
        
num_fail = 0
for idx, qid in enumerate(key_pos): #Rationale dataset need to be in order
    pred = preds[int(idx)]
    ref = targets[int(idx)]
    extract_pred = extract_ans(pred)
    if extract_pred != "FAILED":
        if extract_pred in options:
            extract_pred = options.index(extract_pred)
        else:
            extract_pred = random.choice(range(0,len(options)))
    else:
        num_fail += 1
        extract_pred = random.choice(range(len(options))) # random choose one option
    results_ans[str(qid)] = extract_pred
    results_rationale[str(qid)] = pred
    results_reference[str(qid)] = ref

scores = get_scores(results_ans, results_rationale, results_reference, "data/scienceqa/data/problems.json")
preds = [pred.strip() for pred in preds]
output_data = {
        "num_fail": num_fail,
        "scores": scores,
        "preds": preds,
        "labels": targets}

with open("Saving/ans_eval.json", "w") as writer:
    writer.write(json.dumps(output_data, indent=4))