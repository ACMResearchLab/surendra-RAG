from transformers import AutoTokenizer, AutoModel
import torch

from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained('mchochlov/codebert-base-cd-ft')
model = AutoModel.from_pretrained('mchochlov/codebert-base-cd-ft')

model_2 = SentenceTransformer('mchochlov/codebert-base-cd-ft')


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sentensize(text: str) -> float:

    sentence = [text]

    encoded_input = tokenizer(sentence, padding=True,
                              truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    return sentence_embeddings


cos = torch.nn.CosineSimilarity(dim=1)


def library_thing(text: str, text2: str):

    array = [text]
    array2 = [text2]
    embeddings1 = model_2.encode(array)
    embeddings2 = model_2.encode(array2)

    embeddings1 = torch.from_numpy(embeddings1)
    embeddings2 = torch.from_numpy(embeddings2)
   
    # print(embeddings2)
    # print(embeddings2.shape)
    #cos(embeddings1, embeddings2)
    # print(embeddings1)
    # print(type(embeddings1))
    return cos(embeddings1, embeddings2).item()



def cos_sim(text1: str, text2: str):
    embeddings1 = sentensize(text1)
    embeddings2 = sentensize(text2)

    tensors1 = torch.split(embeddings1, split_size_or_sections=1, dim=0)
    tensors2 = torch.split(embeddings2, split_size_or_sections=1, dim=0)

    return cos(tensors1[0], tensors2[0]).item()
