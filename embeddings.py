from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

#pad it with 0's
#pad it with 1's

nl_tokens = tokenizer.tokenize("return larger number")
print(f"tokens: {nl_tokens}")
code_tokens_1 = tokenizer.tokenize(
    """
    def max(a,b):
        if a > b:
            return a

        return b


            """)
code_tokens_2 = tokenizer.tokenize(
    " def max(a,b): if a>b: return a else return b")
print(f"code tokens 1: {code_tokens_1}")
print(f"code tokens 2: {code_tokens_2}")


def myTokenize(pl_tokens, nl_tokens):
    return [tokenizer.cls_token]+nl_tokens + \
        [tokenizer.sep_token]+pl_tokens+[tokenizer.eos_token]


thing1 = myTokenize(code_tokens_1, nl_tokens)
thing2 = myTokenize(code_tokens_2, nl_tokens)


print(f"tokens 1: {thing1}")
print(f"tokens 2: {thing2}")

tokens_1_ids = tokenizer.convert_tokens_to_ids(thing1)
tokens_2_ids = tokenizer.convert_tokens_to_ids(thing2)

print(f"token_ids: {tokens_1_ids}")
print(f"token_ids: {tokens_2_ids}")

context_embeddings_1 = model(torch.tensor(tokens_1_ids)[None, :])[0]
context_embeddings_2 = model(torch.tensor(tokens_2_ids)[None, :])[0]

print(f"context_embeddings_1 : {context_embeddings_1}")
print(f"context_embeddings_2: {context_embeddings_2}")
print(f"torch size_1: {context_embeddings_1.size()}")
print(f"torch size_2: {context_embeddings_2.size()}")

cos = torch.nn.CosineSimilarity(dim=2)

new_context_embeddings_2 = F.pad(input=context_embeddings_2, pad=(0, 0 ,45, 0), mode='constant', value=0)

print(f"torch size_2_padded: {new_context_embeddings_2.size()}")


x = cos(context_embeddings_1, new_context_embeddings_2)





print(f"cosine similarity_norm:{torch.linalg.vector_norm(input=x)}")
print(f"cosine similarity size: {x.size()}")
