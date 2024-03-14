import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from example_functions import functions


# use two different models

tokenizer_1 = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_1 = AutoModel.from_pretrained("microsoft/codebert-base")
model_2 = SentenceTransformer('mchochlov/codebert-base-cd-ft')

cos = torch.nn.CosineSimilarity(dim=1)
testset = functions


def model_1_cosine_similarities(text1: str, text2: str):

    # Mean Pooling - Take attention mask into account for correct averaging
    def sentensize(text: str):
        # Sentences we want sentence embeddings for
        sentences = [text]

# Load model from HuggingFace Hub

# Tokenize sentences
        encoded_input = tokenizer_1(sentences, padding=True,
                                    truncation=True, return_tensors='pt')

# Compute token embeddings
        with torch.no_grad():
            model_output = model_1(**encoded_input)

# Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings

    embeddings_1 = sentensize(text1)
    embeddings_2 = sentensize(text2)

    return cos(embeddings_1, embeddings_2).item()


def model_2_cosine_similarities(text1: str, text2: str):
    def embeddize(text: str): return model_2.encode([text])

    embeddings_1 = torch.from_numpy(embeddize(text1))
    embeddings_2 = torch.from_numpy(embeddize(text2))

    return cos(embeddings_1, embeddings_2).item()


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


query = """
def function_102931(nums):
    if not nums:
        return None
    return max(nums)

"""
model_1_similarities = []
for x in testset:
    model_1_similarities.append(model_1_cosine_similarities(query, x))

a = np.array(model_1_similarities)

# Find 4 top indexes
x = 5
ind = np.argpartition(a, x*-1)[x*-1:]

# Sort them
ind = ind[np.argsort(a[ind])][::-1]


print(f"Query: {query}")


print("+++++++++++++++++ codebert base mean pooling similarities ++++++++++++++ ")
for i in ind:
    print(testset[i])
    print(f"similarity val: {model_1_similarities[i]}")

model_2_similarities = []
for x in testset:
    model_2_similarities.append(model_2_cosine_similarities(query, x))

print("+++++++++++++++++ sentence embeddings similarities ++++++++++++++ ")

a = np.array(model_2_similarities)

# Find 4 top indexes
# ind = np.argpartition(a, -4)[-4:]
x = 5
ind = np.argpartition(a, x*-1)[x*-1:]
# Sort them
ind = ind[np.argsort(a[ind])][::-1]

for i in ind:
    print(testset[i])
    print(f"similarity val: {model_2_similarities[i]}")
