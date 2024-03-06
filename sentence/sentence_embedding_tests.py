from main_sentence_embeddings import cos_sim




m = "add two numbers"

o = """

def add_numbers(a, b):
    return a + b

"""

p = """
def generate_random_sentence():
    subjects = ['I', 'You', 'He', 'She', 'They']
    verbs = ['run', 'jump', 'dance', 'sing', 'laugh']
    objects = ['cat', 'dog', 'ball', 'tree', 'car']

    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    return f"{subject} {verb}s with a {obj}."

"""





print(cos_sim(m, p))
print(cos_sim(m, o))



