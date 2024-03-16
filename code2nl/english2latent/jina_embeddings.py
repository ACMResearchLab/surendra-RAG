from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from ..code2english import gemma, codet5p
from ..code2english import gemma, codet5p

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en",  # switch to en/zh for English or Chinese
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024


def get_embeddings(text: str):
    embeddings = model.encode([text])
    return embeddings
