
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# use two different models

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/codebert-base")

code = """
def add_numbers(a, b):
    return a + b
"""

# Tokenize the code
inputs = tokenizer.encode(
    "translate code to natural language: " + code, return_tensors="pt")

# Generate natural language description
outputs = model.generate(inputs, max_length=100,
                         num_beams=10, early_stopping=True)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated natural language description:")
print(decoded_output)
