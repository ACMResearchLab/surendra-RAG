
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


def padded_0_cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    cos = torch.nn.CosineSimilarity(dim=2)

    # query
    tensor_1_vectors = torch.split(vec1, split_size_or_sections=1,  dim=1)
    # docs
    tensor_2_vectors = torch.split(vec2, split_size_or_sections=1,  dim=1)

    print(len(tensor_1_vectors))
    print(len(tensor_2_vectors))
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

    print(final_cos)

    # Get the dimensions of the tensors
    # dim_vec1 = vec1.shape
    # dim_vec2 = vec2.shape
    # print(dim_vec2)
    # print(dim_vec1)

    # Perform subtraction of dimensions
    # difference = [dim1 - dim2 for dim1, dim2 in zip(dim_vec1, dim_vec2)]

    # if difference[1] < 0:
    #     vec1 = F.pad(input=vec1, pad=(
    #         0, 0, -1*difference[1], 0), mode='constant', value=0)
    #
    # elif difference[1] > 0:
    #     vec2 = F.pad(input=vec2, pad=(
    #         0, 0, difference[1], 0), mode='constant', value=0)
    #
    # else:
    #     None

    # cos_vector = cos(vec1, vec2)
    # return torch.mean(cos_vector)
    # return cos_vector

    return 1


natural_language = "return larger number"
nl_tokens = tokenizer.tokenize(natural_language)

gibberish_natural_language = "fai;uhdlahdlawhdiwaulhdiauhdia"
gibb_nl_tokens = tokenizer.tokenize(natural_language)
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
embeds_12 = pl_embedding(code_1, gibb_nl_tokens)
embeds_2 = pl_embedding(code_2, nl_tokens)
embeds_3 = pl_embedding(code_3, nl_tokens)
embeds_4 = pl_embedding(code_4, nl_tokens)


new_embeds_1 = pl_embedding_without_nl(code_1)

new_embeds_2 = pl_embedding_without_nl(code_2)

new_embeds_3 = pl_embedding_without_nl(code_3)
print("the same")
c1 = padded_0_cos_similarity(new_embeds_1, new_embeds_1)
print("semantically similar things")
c1 = padded_0_cos_similarity(new_embeds_1, new_embeds_2)

print("random stuff")
c1 = padded_0_cos_similarity(new_embeds_1, new_embeds_3)



# print(embeds_1)
# print(embeds_2)
# print(embeds_1.size())
# print(embeds_2.size())
# c1 = padded_0_cos_similarity(embeds_1, embeds_2)
# c2 = padded_0_cos_similarity(embeds_1, embeds_3)
# c3 = padded_0_cos_similarity(embeds_3, embeds_4)
# c4 = padded_0_cos_similarity(embeds_1, embeds_1)
# c12 = padded_0_cos_similarity(embeds_1, embeds_12)
#
#
# print(
#     f"cosine similarity between {code_1} and {code_1} \n with {gibberish_natural_language} \n cosine similarity = {c12} ")
#
#
# print(f"cosine similarity between {code_1} and {code_2}: {c1} ")
# print(f"cosine similarity between {code_1} and {code_3}: {c2} ")
# print(f"cosine similarity between {code_3} and {code_4}: {c3} ")
# print(f"cosine similarity between {code_1} and {code_1}: {c4} ")
