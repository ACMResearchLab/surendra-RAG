
from transformers import RobertaTokenizer, T5ForConditionalGeneration
tokenizer = RobertaTokenizer.from_pretrained(
    'Salesforce/codet5-base-multi-sum')
model = T5ForConditionalGeneration.from_pretrained(
    'Salesforce/codet5-base-multi-sum')


def code_2_english(text: str):

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=100)
    english = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return english
