from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")

model = AutoModelForTokenClassification.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")

text = (
    "“Well, Prince, so Genoa and Lucca are now just family estates of the"
    "Buonapartes. But I warn you, if you don’t tell me that this means war,"
    "if you still try to defend the infamies and horrors perpetrated by that"
    "Antichrist—I really believe he is Antichrist—I will have nothing"
    "more to do with you and you are no longer my friend, no longer my"
    "‘faithful slave,’ as you call yourself! But how do you do? I see I"
    "have frightened you—sit down and tell me all the news.”"
    ).lower()

pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
output = pipe(text)

data = pd.DataFrame(output)
print(data)
