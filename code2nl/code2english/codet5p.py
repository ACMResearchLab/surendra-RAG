
from transformers import RobertaTokenizer, T5ForConditionalGeneration

old_text = """

import random as r

def s():
    return r.random()

print(s())

"""

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    text = """
import random as r

def s():
    return r.random()

print(s())

    """

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=100)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    # this prints: "Convert a SVG string to a QImage."
