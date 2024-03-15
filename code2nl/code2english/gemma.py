from transformers import AutoTokenizer, AutoModelForCausalLM
# pip install accelerate

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", device_map="auto",)

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
    
    print(english)
    print("NEW!!!!!!!!!!!!!!!!!1")
    return english
