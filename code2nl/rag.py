from .english2latent import jina_embeddings
import torch
import heapq

from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1, eps=1e-6)

'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, model_name: str):
    embeddings = []
    for x in contexts:
        np_embedding = jina_embeddings.get_embeddings(x, model_name)
        ttensor_embedding = torch.from_numpy(np_embedding)
        embeddings.append(ttensor_embedding)

    similarities = []

    query_embedding = torch.from_numpy(
        jina_embeddings.get_embeddings(query, model_name))

    for x in embeddings:
        similarities.append(cos(query_embedding, x).item())

# Indices of N largest elements in list
# using heapq.nlargest()
    res = [similarities.index(i) for i in heapq.nlargest(n, similarities)]

# printing result
    # print("Indices list of max N elements is : " + str(res))
    #
    # for x in res:
    #     print(similarities[x])
    print(f"original query: {query} \n")

    print("RAG Results")

    for x in res:
        print(f"{contexts[x]} \n")


# print(type(embeddings))
#
# print(type(embeddings[0]))
# print(embeddings[0].shape)
