
from sentence_transformers import SentenceTransformer
from example_functions import functions
code_fragments = functions

model = SentenceTransformer('mchochlov/codebert-base-cd-ft')

embeddings = model.encode(code_fragments)


print(embeddings)


def get_embeddings(text: str):
    input_arr = [text]
    return model.encode(input_arr)[0]
