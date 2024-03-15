import codebert
import codet5p


'''TODO: you get a chunk of functions, and you have to take cosine similarity
between all of them and return the most similar functions



'''


def get_n_highest_similar_to(query: str, contexts: list, n: int, model_name: str):

    model = None

    match model_name:
        case "codet2p":
            model = codet5p
        case "codebert":
            model = codebert
        case _:
            print("choose codet2p or codebert")
            return

    embeddings = None

    for x in contexts:
        embeddings.append(model.get_embeddings(x))

    similarities = None

    print(type(embeddings))
