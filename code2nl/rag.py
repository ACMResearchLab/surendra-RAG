from .english2latent import jina_embeddings
import torch


'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, model_name: str):
    embeddings = []
    for x in contexts:
        np_embedding = jina_embeddings.get_embeddings(x, model_name)
        ttensor_embedding = torch.from_numpy(np_embedding)
        embeddings.append(ttensor_embedding)

    similarities = None
    
    # query_embedding = 
    print(type(embeddings))

    print(type(embeddings[0]))
    print(embeddings[0].shape)
