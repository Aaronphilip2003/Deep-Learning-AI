from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# Load your model
model = KeyedVectors.load_word2vec_format('./glove.6B.50d.word2vec.txt', binary=False)

# Step 1: Get vectors
vec_king = model['king']
vec_man = model['man']
vec_woman = model['woman']

# Step 2: Compute difference
difference = vec_king - vec_man
print("🔍 Difference vector (king - man):")
print(difference)

# Step 3: Add difference to 'woman'
result_vector = vec_woman + difference

# Step 4: Find closest words to result vector
similar_words = model.similar_by_vector(result_vector, topn=5)

print("\n🔎 Words closest to (king - man + woman):")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")



