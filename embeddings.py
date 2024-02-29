from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# pad it with 0's
# pad it with 1's


def pl_embedding(pl: str, nl_tokens):
    pl_tokens = tokenizer.tokenize(pl)

    tokens = [tokenizer.cls_token]+nl_tokens + \
        [tokenizer.sep_token]+pl_tokens+[tokenizer.eos_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)

    embeddings = model(torch.tensor(ids)[None, :])[0]

    return embeddings


def padded_0_cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    cos = torch.nn.CosineSimilarity(dim=2)
    # print(vec1.shape - vec2.shape)
    # Get the dimensions of the tensors
    dim_vec1 = vec1.shape
    dim_vec2 = vec2.shape

# Perform subtraction of dimensions
    difference = [dim1 - dim2 for dim1, dim2 in zip(dim_vec1, dim_vec2)]
    if difference[1] < 0:
        vec1 = F.pad(input=vec1, pad=(
            0, 0, -1*difference[1], 0), mode='constant', value=0)

    elif difference[1] > 0:
        vec2 = F.pad(input=vec2, pad=(
            0, 0, difference[1], 0), mode='constant', value=0)

    else:
        None

    cos_vector = cos(vec1, vec2)
    return torch.mean(cos_vector)
    # return cos_vector


natural_language = "return larger number"
nl_tokens = tokenizer.tokenize(natural_language)

code_1 = """
def max(a,b):
    if a > b:
        return a

    return b


        """

code_2 = "def max(a,b): if a>b: return a else return b"

code_3 = "edyuasefghliahefklashfjk,sdhvjfdhgkjghxdjkhgfsjgjyypkhsdgjh"
code_4 = "flyuraglkjfhg arsjkfhvaskbrhgskrjghvjkaernghvagvhkashgnvka    "

embeds_1 = pl_embedding(code_1, nl_tokens)
embeds_2 = pl_embedding(code_2, nl_tokens)
embeds_3 = pl_embedding(code_3, nl_tokens)
embeds_4 = pl_embedding(code_4, nl_tokens)
# print(embeds_1)
# print(embeds_2)
# print(embeds_1.size())
# print(embeds_2.size())
c1 = padded_0_cos_similarity(embeds_1, embeds_2)
c2 = padded_0_cos_similarity(embeds_1, embeds_3)
c3 = padded_0_cos_similarity(embeds_3, embeds_4)
c4 = padded_0_cos_similarity(embeds_1, embeds_1)


print(f"cosine similarity between {code_1} and {code_2}: {c1} ")
print(f"cosine similarity between {code_1} and {code_3}: {c2} ")
print(f"cosine similarity between {code_3} and {code_4}: {c3} ")
print(f"cosine similarity between {code_1} and {code_1}: {c4} ")
