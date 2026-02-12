📚 Multimodal Book Genre Classification & Clustering

University of Houston–Clear Lake | Aug 2024 – Dec 2024

A scalable multimodal machine learning pipeline for book genre prediction, clustering, and segmentation using both book cover images and textual metadata (titles).

This project combines Computer Vision + NLP to automatically organize and segment large-scale book catalogs for recommendation systems and digital libraries.

🚀 Project Overview

Modern online bookstores contain millions of listings with inconsistent metadata. This project builds a multimodal deep learning system that:

Predicts genres using book cover images + titles

Performs unsupervised clustering for large-scale segmentation

Evaluates separability using silhouette analysis

Produces interpretable groupings for recommendation pipelines

📊 Datasets
1️⃣ BookCover30 (Supervised Classification)

57,000 book covers

30 genre classes

90% training / 10% testing split

Used for multimodal genre prediction

2️⃣ Book32 (Large-Scale Mining)

207,572 books

32 classes

Used for clustering & scalable segmentation experiments

🧠 Methodology
🔹 1. Data Preprocessing

Text (Titles)

Tokenization

Vocabulary indexing

Padding sequences

Embedding layer preparation

Images (Covers)

Resizing & normalization

Feature extraction using pretrained CNN (VGG16)

🔹 2. Multimodal Deep Learning Architecture

The classification model combines:

🖼 VGG16 visual embeddings

📝 Embedding + LSTM for titles

🔗 Concatenation layer

🎯 Dense layers for genre prediction

Image Input → VGG16 → Visual Features
Text Input → Embedding → LSTM → Text Features
                   ↓
            Concatenation
                   ↓
              Dense Layers
                   ↓
            Genre Prediction


Framework: TensorFlow / Keras

🔹 3. Unsupervised Clustering Pipeline

For large-scale mining:

Extracted deep feature embeddings

Reduced dimensionality using PCA

Applied KMeans clustering

Evaluated cluster quality using Silhouette Score

This enabled interpretable segmentation beyond labeled categories.

📈 Key Results

Multimodal fusion outperformed unimodal (image-only or text-only) models

Clear genre separability observed in PCA-reduced embedding space

Silhouette scores validated clustering stability

Demonstrated scalability to 200k+ book dataset

🛠️ Tech Stack

Python

TensorFlow / Keras

scikit-learn

NumPy / Pandas

Matplotlib / Seaborn

PCA & KMeans

VGG16 (Transfer Learning)
