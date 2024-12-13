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
    rat_f=[w if not w.lower() in filtered_sentence else "_" for w in rat]
    rationale_sentence[key] = ".".join(rat_sentence[:end])
    rationale_remove[key] = " ".join(rat_f)

with open("Saving/rationale_masked_sentence.json", "w") as writer:
    writer.write(json.dumps(rationale_sentence, indent=4))

with open("Saving/rationale_masked_words.json", "w") as writer:
    writer.write(json.dumps(rationale_remove, indent=4))
