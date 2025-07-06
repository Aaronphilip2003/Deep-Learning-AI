import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings from GloVe .txt file, skipping malformed lines
def load_embeddings(file_path, dim=50):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            values = parts[1:]
            if len(values) != dim:
                continue  # Skip malformed vectors
            vector = np.array(values, dtype=np.float32)
            embeddings[word] = vector
    return embeddings

# Find top-N most similar words to a given vector
def most_similar(embeddings, vector, topn=5, skip=set()):
    words = []
    vectors = []
    for word, vec in embeddings.items():
        if word not in skip:
            words.append(word)
            vectors.append(vec)

    vectors = np.stack(vectors)
    sims = cosine_similarity([vector], vectors)[0]
    top_indices = np.argsort(-sims)[:topn]
    return [(words[i], sims[i]) for i in top_indices]

# Load the GloVe embeddings
embeddings = load_embeddings('glove.6B.50d.word2vec.txt', dim=50)

# Get vectors for selected words
vec_king = embeddings['king']
vec_man = embeddings['man']
vec_woman = embeddings['woman']
vec_cat = embeddings['cat']
vec_cats = embeddings['cats']
vec_dog = embeddings['dog']

# Vector arithmetic
plural_diff = vec_cats - vec_cat
difference = vec_king - vec_man
result_vector = vec_woman + difference
result_vector_plural = vec_dog + plural_diff

# Find similar words
print("\n🔎 Words closest to (king - man + woman):")
for word, score in most_similar(embeddings, result_vector, topn=5, skip={'king', 'man', 'woman'}):
    print(f"{word}: {score:.4f}")

print("\n🔎 Plural Vector closest to dog:")
for word, score in most_similar(embeddings, result_vector_plural, topn=5, skip={'dog', 'cat', 'cats'}):
    print(f"{word}: {score:.4f}")
