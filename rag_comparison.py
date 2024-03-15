# from . import code2nl
# from . import code2latent

from code2nl import rag as code2nl_rag
from code2latent import rag as code2lat_rag

from random_functions import test_functions

query = """
def _fE2jR5(lst):
    return [x for x in lst if x % 2 != 0] 
"""

code2nl_rag.get_n_highest_similar_to(query, test_functions, 5, "gemma")

# TODO: do this
# code2lat_rag.get_n_highest_similar_to(
#     "heakdjwajdaw", test_functions, 5, "codet5p")
#
