from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en", # switch to en/zh for English or Chinese
    trust_remote_code=True
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    'How is the weather today?',
    'What is the current weather like today?',
    'what is wrong with you dude, i hate you',
    'i love you pookie',
    'p9ewf8ouyahofuhawoughfaisuhfisuzhfasehfilshifuhsaeilufszifh'
])
print(cos_sim(embeddings[0], embeddings[1]))
print(cos_sim(embeddings[2], embeddings[3]))
print(cos_sim(embeddings[0], embeddings[4]))
