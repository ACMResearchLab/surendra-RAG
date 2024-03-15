from .english2latent import jina_embeddings


'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, model_name: str):
    embeddings = []
    for x in contexts:
        embeddings.append(jina_embeddings.get_embeddings(x, model_name))

    similarities = None

    print(type(embeddings))


    print(type(embeddings[0]))
    print(embeddings[0].shape)
