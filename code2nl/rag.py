from .english2latent import jina_embeddings
import torch
import heapq

from .code2english import gemma, codet5p

from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1, eps=1e-6)

'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, c2e_model_name: str, e2l_model_name):

    context_english_descriptions = []
    # TODO: get code 2 english

    for x in contexts:

        match c2e_model_name:
            case "codet5p":
                english = codet5p.code_2_english(x)
                # print(english)
            case "gemma":
                english = gemma.convert_2_english(x)
                # print("gemma is only on big boy")
            case "codetrans":
                print("codeTrans is unsupported")
                return
    print(
        f"length of english descriptions {len(context_english_descriptions)}")
    embeddings = []
    for x in context_english_descriptions:
        np_embedding = None
        match e2l_model_name:
            case "jina":
                np_embedding = jina_embeddings.get_embeddings(x)

            case "bge":

                print("unimplemented")
                return

            case _:
                print("unimplemented")
                return

        ttensor_embedding = torch.from_numpy(np_embedding)
        embeddings.append(ttensor_embedding)

    print(f"length of embeddings {len(embeddings)}")

    similarities = []

    query_embedding = None

    match e2l_model_name:
        case "jina":
            query_embedding = jina_embeddings.get_embeddings(query)

        case "bge":

            print("unimplemented")
            return

        case _:
            print("unimplemented")
            return
    query_embedding = torch.from_numpy(query_embedding)
    print(f"query embedding length {query_embedding}")

    for x in embeddings:
        similarities.append(cos(query_embedding, x).item())

    print(len(similarities))
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
