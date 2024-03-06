from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained('mchochlov/codebert-base-cd-ft')
model = AutoModel.from_pretrained('mchochlov/codebert-base-cd-ft')


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

    return sentence_embeddings[0]
