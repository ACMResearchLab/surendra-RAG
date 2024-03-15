from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from ..code2english import gemma, codet5p
from ..code2english import codet5p

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en",  # switch to en/zh for English or Chinese
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024


def get_embeddings(text: str, model_name: str):
    english = None
    match model_name:
        case "codet5p":
            english = codet5p.code_2_english(text)
            # print(english)
        case "gemma":
            # english = gemma.convert_2_english(text)
            print("gemma is only on big boy")
            return
        case "codetrans":
            print("codeTrans is unsupported")
            return

    embeddings = model.encode([english])
    return embeddings
