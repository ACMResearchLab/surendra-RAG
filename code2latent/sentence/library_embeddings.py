from sentence_transformers import SentenceTransformer
from example_functions import functions
code_fragments = functions

model_2 = SentenceTransformer('mchochlov/codebert-base-cd-ft')

embeddings = model_2.encode(code_fragments)



print(embeddings)
