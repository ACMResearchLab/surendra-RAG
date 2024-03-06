from main_sentence_embeddings import cos_sim
from example_functions import functions
import numpy as np


# feel free to change
query = "give me the slope intercept form of a function"


def rag(query: str, contexts: list):

    print(f"Query: {query}")

    similaritys = []
    for x in contexts:
        similaritys.append(cos_sim(query, x))

    a = np.array(similaritys)

    # Find 4 top indexes
    ind = np.argpartition(a, -4)[-4:]

    # Sort them
    ind = ind[np.argsort(a[ind])][::-1]

    for i in ind:
        print(contexts[i])
        print(f"similarity val: {similaritys[i]}")


rag(query, functions)
