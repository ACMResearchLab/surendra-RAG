# from . import code2nl
# from . import code2latent

from code2nl import rag as code2nl_rag
from code2latent import rag as code2lat_rag

from random_functions import test_functions


code2nl_rag.get_n_highest_similar_to(
    "heakdjwajdaw", test_functions, 5, "codet5p")
code2lat_rag.get_n_highest_similar_to(
    "heakdjwajdaw", test_functions, 5, "codet5p")

