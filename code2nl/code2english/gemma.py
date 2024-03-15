from transformers import AutoTokenizer, AutoModelForCausalLM
# pip install accelerate

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", device_map="auto",)

input_text = """
convert this code into text and dont say anything else. dont print the query

import random as r

def s():
    return r.random()

print(s())


"""


def convert_2_english(text: str):

    input_ids = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_length=100)
    english = tokenizer.decode(outputs[0])

    return english
