
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# pad it with 0's
# pad it with 1's
# does this change the difference?


def pl_embedding_without_nl(pl: str):
    pl_tokens = tokenizer.tokenize(pl)

    tokens = [tokenizer.cls_token] + \
        pl_tokens+[tokenizer.eos_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)

    embeddings = model(torch.tensor(ids)[None, :])[0]

    return embeddings


def pl_embedding(pl: str, nl_tokens):
    pl_tokens = tokenizer.tokenize(pl)

    tokens = [tokenizer.cls_token]+nl_tokens + \
        [tokenizer.sep_token]+pl_tokens+[tokenizer.eos_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)

    embeddings = model(torch.tensor(ids)[None, :])[0]

    return embeddings


def kevin_cos_sim(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    cos = torch.nn.CosineSimilarity(dim=2)

    # query
    tensor_1_vectors = torch.split(vec1, split_size_or_sections=1,  dim=1)
    # docs
    tensor_2_vectors = torch.split(vec2, split_size_or_sections=1,  dim=1)

    final_cos = 0
    for x in range(0, len(tensor_1_vectors)):

        vec1 = tensor_1_vectors[x]
        cos_sums_inner = 0
        max = -1
        for y in range(0, len(tensor_2_vectors)):
            vec2 = tensor_2_vectors[y]

            cos_vector = cos(vec1, vec2)

            cos_value = cos_vector[0][0].item()

            if cos_value > max:
                max = cos_value

            cos_sums_inner += cos_vector[0][0].item()

        # print(max)

        cos_sums_inner = max
        # print(x)
        final_cos += cos_sums_inner
    # print(len(tensor_1_vectors))
    final_cos = final_cos / len(tensor_1_vectors)

    return final_cos


