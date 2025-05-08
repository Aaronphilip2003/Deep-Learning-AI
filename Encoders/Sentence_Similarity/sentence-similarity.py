from sentence_transformers import SentenceTransformer
from tabulate import tabulate

input_sentences = [
    "The sun is shining",
    "The sun is bright",
    "I love learning computer science and programming in Python",
    "I want to learn about Encoders in computer science",
    "I love football",
    "I love exercise",
]

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the input sentences
sentence_embeddings = model.encode(input_sentences)

# Calculate the cosine similarity between the sentences
similarities = model.similarity(sentence_embeddings, sentence_embeddings)
print(similarities.shape)

# Create shorter labels for the sentences
short_labels = [
    "Sun shining",
    "Sun bright",
    "CS & Python",
    "Encoders",
    "Football",
    "Exercise"
]

# Print top similar pairs (excluding self-similarity)
print("\nSentence Similarities (sorted by similarity):")
print("-" * 45)
pairs = []
for i in range(len(input_sentences)):
    for j in range(i+1, len(input_sentences)):
        pairs.append((short_labels[i], short_labels[j], similarities[i][j]))

def get_similarity_score(pair):
    """Return the similarity score from a pair tuple (label1, label2, similarity)"""
    return pair[2]

# Sort pairs by similarity score in descending order
pairs.sort(key=get_similarity_score, reverse=True)

# Convert pairs to table format
table_data = []
headers = ["Sentence 1", "Sentence 2", "Similarity"]
for first_label, second_label, score in pairs:
    table_data.append([first_label, second_label, f"{score:.3f}"])

# Print the table
print("\nSentence Similarity Results:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))