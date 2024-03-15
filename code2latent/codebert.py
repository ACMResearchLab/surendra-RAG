
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('mchochlov/codebert-base-cd-ft')

def get_embeddings(text: str):
    input_arr = [text]
    return model.encode(input_arr)[0]
