from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
from tabulate import tabulate
import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import MDS, TSNE

data = load_dataset("quora",trust_remote_code=True)

# Create a dictionary to store question pairs and their similarities
question_pairs = {}

for i, item in enumerate(data['train']):
    question_pairs[f"pair_{i+1}"] = {
        'questions': item['questions']['text'],
        'similarity': None  # Will be filled later
    }
    if i >= 10:  # Store first 5 pairs
        break

model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Calculate similarities and store them in the dictionary
for pair_id, pair_data in question_pairs.items():
    # Encode both questions in the pair
    embeddings = model.encode(pair_data['questions'])
    # Calculate similarity using sentence-transformers util
    similarity = model.similarity(embeddings[0], embeddings[1])
    pair_data['similarity'] = float(similarity)

# Create a DataFrame to store the results
results = pd.DataFrame({
    'Question 1': [pair['questions'][0] for pair in question_pairs.values()],
    'Question 2': [pair['questions'][1] for pair in question_pairs.values()],
    'Similarity': [f"{pair['similarity']:.4f} ({pair['similarity'] * 100:.2f}%)" for pair in question_pairs.values()]
})

# Store raw similarities for statistics
raw_similarities = [pair['similarity'] for pair in question_pairs.values()]

# Add styling and better formatting
pd.set_option('display.max_colwidth', None)
styled_results = results.style\
    .set_properties(**{'text-align': 'left'})\
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), 
                                   ('color', '#333'),
                                   ('font-weight', 'bold'),
                                   ('padding', '10px')]},
        {'selector': 'td', 'props': [('padding', '8px')]}
    ])

# Print a nice header
print("\n" + "="*80)
print("Question Pair Similarity Analysis")
print("="*80 + "\n")

# Display the results
print(styled_results.to_string())

# Print summary statistics
print("\nSummary Statistics:")
print("-"*40)
print(f"Average Similarity: {sum(raw_similarities)/len(raw_similarities):.4f} ({sum(raw_similarities)/len(raw_similarities)*100:.2f}%)")
print(f"Highest Similarity: {max(raw_similarities):.4f} ({max(raw_similarities)*100:.2f}%)")
print(f"Lowest Similarity: {min(raw_similarities):.4f} ({min(raw_similarities)*100:.2f}%)")

# Only keep the 2D visualization section
# Get embeddings for all questions
all_questions = []
for pair in question_pairs.values():
    all_questions.extend(pair['questions'])
all_embeddings = model.encode(all_questions)

# Use MDS to reduce dimensionality to 3D while preserving distances
mds = MDS(n_components=3, random_state=42)
embeddings_3d = mds.fit_transform(all_embeddings)

# Create color list with valid RGB values
colors = []
for i in range(len(question_pairs)):
    # Create colors using HSL color space for better distribution
    hue = (i * 360 // len(question_pairs)) % 360
    colors.extend([f'hsl({hue}, 70%, 50%)'] * 2)  # Using HSL colors instead

# Create the 3D scatter plot
scatter = go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers+text',
    marker=dict(
        size=10,
        color=colors,
        opacity=0.8
    ),
    text=[f'Q{i+1}<br>{q[:50]}...' for i, q in enumerate(all_questions)],
    hoverinfo='text',
    textposition='top center'
)

# Create lines connecting pairs
lines = []
for i in range(0, len(embeddings_3d), 2):
    lines.append(
        go.Scatter3d(
            x=[embeddings_3d[i, 0], embeddings_3d[i+1, 0]],
            y=[embeddings_3d[i, 1], embeddings_3d[i+1, 1]],
            z=[embeddings_3d[i, 2], embeddings_3d[i+1, 2]],
            mode='lines',
            line=dict(color=colors[i], width=2, dash='dash'),
            showlegend=False
        )
    )

# Create the figure
fig = go.Figure(data=[scatter] + lines)

# Update the layout
fig.update_layout(
    title='3D Visualization of Question Embeddings<br>(Connected pairs have similar meanings)',
    scene=dict(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        zaxis_title='Dimension 3',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    showlegend=False,
    margin=dict(l=0, r=0, t=30, b=0)
)

# Show the plot
fig.show()