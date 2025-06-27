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

vec_cat=model['cat']
vec_cats=model['cats']
vec_dog=model['dog']
plural_diff= vec_cats-vec_cat

# Step 2: Compute difference
difference = vec_king - vec_man
# Step 3: Add difference to 'woman'
result_vector = vec_woman + difference

result_vector_plural=vec_dog+plural_diff

# Step 4: Find closest words to result vector
similar_words = model.similar_by_vector(result_vector, topn=5)

print("\n🔎 Words closest to (king - man + woman):")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")


similar_words = model.similar_by_vector(result_vector_plural, topn=5)

print("\n🔎 Plural Vector closest to dog:")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")