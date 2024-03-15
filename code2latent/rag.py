from .codebert import get_embeddings as codebert_embeddings
from .codet5p import get_embeddings as codet5p_embeddings

'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, model_name: str):

    embeddings_function = None

    match model_name:
        case "codet2p":
            embeddings_function = codet5p_embeddings
        case "codebert":
            embeddings_function = codebert_embeddings
        case _:
            print("choose codet2p or codebert")
            return

    embeddings = []

    for x in contexts:
        embeddings.append(embeddings_function(x))

    similarities = None

    print(type(embeddings))
