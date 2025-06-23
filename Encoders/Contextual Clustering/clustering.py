import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import plotly.io as pio
pio.renderers.default = 'browser'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'tables.json')

# Read and parse the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Create dictionary of table_name: {'columns': [column_names], 'description': table_description, 'domain': domain}
table_dict = {}
for table in data['tables']:
    table_dict[table['name']] = {
        'columns': [col['name'] for col in table['columns']],
        'description': table['description'],
        'domain': table['domain']
    }

# print total number of tables
print(f"Total number of tables: {len(table_dict)}")

# Print the dictionary
print("Table Dictionary:")
for table_name, info in table_dict.items():
    print(f"{table_name}:{info['columns']}")
    print(f"Description: {info['description']}")
    print(f"Domain: {info['domain']}\n")

# Now we need to cluster the tables in the dictionary based on the description

model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Embed the descriptions
descriptions = [info['description'] for info in table_dict.values()]
table_names = list(table_dict.keys())
domains = [info['domain'] for info in table_dict.values()]

embeddings = model.encode(descriptions, convert_to_tensor=True)

# Convert embeddings to numpy and reduce to 2D using PCA
embeddings_np = embeddings.cpu().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)

# Cluster the embeddings using KMeans
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_np)

# Create a scatter plot with custom hover text
fig = go.Figure()

# Add scatter points
for i in range(len(embeddings_2d)):
    fig.add_trace(go.Scatter(
        x=[embeddings_2d[i, 0]],
        y=[embeddings_2d[i, 1]],
        mode='markers+text',
        name=domains[i],
        text=[table_names[i]],
        textposition="top center",
        hovertemplate=f"Table: {table_names[i]}<br>Domain: {domains[i]}<br>Description: {descriptions[i]}",
        marker=dict(size=10),
        showlegend=True
    ))

# Add circles around clusters
for i in range(n_clusters):
    cluster_points = embeddings_2d[cluster_labels == i]
    if len(cluster_points) > 0:
        center = np.mean(cluster_points, axis=0)
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1)) * 1.2
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center[0] + radius * np.cos(theta)
        circle_y = center[1] + radius * np.sin(theta)
        
        # Add the circle
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            name=f'Cluster {i+1}',
            line=dict(dash='dot'),
            showlegend=True
        ))

# Update layout
fig.update_layout(
    title="Table Clusters based on Description Similarity",
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
    showlegend=True,
    width=1200,
    height=800
)

fig.show()