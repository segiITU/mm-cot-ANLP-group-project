import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
 
nltk.download('stopwords')

problems = json.load(open("data/scienceqa/data/problems.json"))
rationale = json.load(open("Saving/predictions_ans_eval.json"))

rationale_sentence=dict()
rationale_remove=dict()
stop_words = stopwords.words('english') + ["'s", "'ll'", "'ve'","'t", "'m'"]
for key in rationale:
    rat=rationale[key]["generated_rationale"]
    rat=rat.replace(".n",". ")
    rat_sentence = rat.split(".")
    end = -1 if rat_sentence[-1]!="" else -2
    rat=word_tokenize(rat)
    word_tokens=word_tokenize(problems[key]["question"])
    
    for choice in problems[key]["choices"]:
        for word in word_tokenize(choice):
            word_tokens.append(word)
    word_tokens=set(word_tokens)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    rat_f=[]
    count = 0
    for w in rat:
        if w.lower() not in filtered_sentence:
            rat_f.append(w)
        else:
            count+=1
            rat_f.append("_")
    rationale_sentence[key]["generated_rationale"] = ".".join(rat_sentence[:end])
    rationale_remove[key]["generated_rationale"] = " ".join(rat_f)
    rationale_remove[key]["count"] = count

with open("Saving/rationale_masked_sentence.json", "w") as writer:
    writer.write(json.dumps(rationale_sentence, indent=4))

with open("Saving/rationale_masked_words.json", "w") as writer:
    writer.write(json.dumps(rationale_remove, indent=4))
