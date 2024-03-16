from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

delimiter = "STOPTAG"

def strip_after_delimiter(s, delimiter):
    delimiter_index = s.find(delimiter)
    if delimiter_index != -1:
        return s[delimiter_index + len(delimiter):]
    else:
        return s

def convert_2_english(text: str):

    text = f"""
    ONLY convert the following code into the semantic meaning, do NOT say anything else. Your response should be a 1 sentence summary, delimiter shoould not be anywhere in your response\n
   
   CODE START
   {text} 
    CODE END

   {delimiter}

    """

    input_ids = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_length=1000)
    english = tokenizer.decode(outputs[0])

    english = strip_after_delimiter(english, delimiter).strip()

    # print(english)
    # print("NEW!!!!!!!!!!!!!!!!!1")
    return english
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
#
# tokenizer = AutoTokenizer
# print(output)
if __name__ == "__main__":
    text = f"""
    convert this to english 

    def _dH6fJ5(lst):
    return sum(1 for x in lst if x > 0)

    """
    
    input_ids = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_length=1000)
    english = tokenizer.decode(outputs[0])

    print(english)

