# Deep-Learning-AI

A repository dedicated to exploring and implementing Deep Learning and AI concepts with practical applications.

## Table of Contents
- [Project Structure](#project-structure)
- [Projects](#projects)
  - [Sentence Similarity Analysis](#1-sentence-similarity-analysis)
  - [Sentence Embedding Visualization](#2-sentence-embedding-visualization)
- [Getting Started](#getting-started)
- [Future Goals](#future-goals)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

### 1. Sentence Similarity Analysis

This project implements semantic similarity comparison between sentences using transformer-based embeddings.

![Sentence Similarity Analysis](./Encoders/Sentence_Similarity/image.png)

#### Features
- Utilizes SentenceTransformer with 'all-MiniLM-L6-v2' model
- Computes pairwise cosine similarities between input sentences
- Presents results in a clear, tabulated format
- Demonstrates practical application of sentence embeddings

#### Example Use Cases
- Finding similar sentences or documents
- Semantic text matching
- Content-based recommendation
- Duplicate detection

#### Technical Details
- Model: all-MiniLM-L6-v2 (efficient and accurate for similarity tasks)
- Similarity metric: Cosine similarity
- Output format: Grid-formatted table with pairwise similarities

### 2. Sentence Embedding Visualization

This project demonstrates semantic similarity analysis and visualization of text embeddings using state-of-the-art transformer models.

![3D Visualization of Question Embeddings](./Encoders/Embedding%20Viz/image.png)

#### Features
- Uses SentenceTransformer with 'paraphrase-mpnet-base-v2' model for generating embeddings
- Calculates semantic similarity between question pairs
- Provides 3D interactive visualization of embedding spaces
- Demonstrates dimensionality reduction using MDS (Multidimensional Scaling)

#### Key Findings
From the sample output:
- Questions with similar semantic meaning show high similarity scores (80-90%)
- Unrelated questions show very low similarity scores (10-20%)
- The 3D visualization clearly clusters semantically related questions together

#### Technical Implementation
- Sentence embeddings using SentenceTransformer
- 3D visualization using Plotly
- Dimensionality reduction using scikit-learn's MDS
- Dataset: Quora Question Pairs

## Getting Started

## Future Goals

## Contributing

## License