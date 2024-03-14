from main_sentence_embeddings import cos_sim, library_thing
from example_functions import functions
import numpy as np


# feel free to change

query = """
def find_max(nums):
    if not nums:
        return None
    return max(nums)
    """


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


def library_test(query: str, contexts):
    similarities = []

    for x in contexts:
        similarities.append(library_thing(query, x))

    a = np.array(similarities)

    # Find 4 top indexes
    ind = np.argpartition(a, -4)[-4:]

    # Sort them
    ind = ind[np.argsort(a[ind])][::-1]

    for i in ind:
        print(contexts[i])
        print(f"similarity val: {similarities[i]}")

#rag(query, functions)

library_test(query, functions)
