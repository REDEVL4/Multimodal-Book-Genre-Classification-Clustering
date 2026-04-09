# Project Analysis and Scope Extension

## Repository analysis

The repository `REDEVL4/Multimodal-Book-Genre-Classification-Clustering` is currently centered around a single notebook, [BookGenreClassification.ipynb](../BookGenreClassification.ipynb), plus a short README. The notebook combines supervised genre classification and unsupervised clustering in one linear workflow.

### Current strengths

- The project is based on a real and widely reused benchmark: BookCover30 and Book32 from the Uchida Lab dataset.
- It correctly identifies the value of multimodality instead of relying on cover images alone.
- It already connects classification and clustering, which is the right direction for library organization and recommendation use cases.
- It uses transfer learning for images instead of training a visual encoder completely from scratch.

### Current limitations

1. `Notebook-only implementation`

The entire workflow lives in one notebook. That makes reuse, debugging, batch inference, and reproducibility harder.

2. `Hard-coded local paths`

The notebook references paths such as a personal Windows downloads directory. The project cannot be rerun without editing multiple cells.

3. `Dataset drift from Task1`

The linked benchmark task is `BookCover30`, but the notebook actually loads `book32-listing.csv`, then samples about 10,000 rows into a balanced subset. That means the implementation is no longer directly comparable to the official Task1 setup.

4. `Weak recorded evaluation`

The visible notebook outputs show:

- training begins on roughly 9,984 sampled items
- the logged classification report covers only 10 test examples
- reported test accuracy is `0.10`
- reported silhouette score is `0.0687`

These values suggest the current saved notebook results are not yet strong enough for production-like deployment.

5. `Heavy visual head`

The notebook uses `VGG16` with `Flatten()`, which creates a very large dense representation and increases training cost. Modern pooling-based heads are typically more efficient and easier to train.

6. `No operational confidence strategy`

For digital libraries, a model should not be forced to classify every book. It should abstain or request review when confidence is low.

7. `Clustering is not operationalized`

The notebook clusters embeddings, but does not export cluster assignments, prototypes, or a downstream retrieval index that a library system can actually use.

## How the project was extended

The new code turns the prototype into a reusable system with three practical layers:

### 1. Classification

- Uses a modular PyTorch multimodal model instead of notebook-defined layers.
- Combines cover-image features with title and author text.
- Supports image-only and multimodal inference modes.
- Returns top-k labels and confidence scores.

### 2. Review-aware automation

- Adds a confidence threshold.
- Marks uncertain items with `needs_review`.
- Makes the system suitable for semi-automatic cataloging rather than risky full automation.

### 3. Catalog intelligence beyond labels

- Exports fused embeddings.
- Clusters books in the embedding space.
- Identifies cluster prototypes.
- Creates a basis for similarity search and recommendation.

## Why this is a better fit for digital libraries

Libraries need systems that reduce labor without reducing trust. A practical genre-assignment pipeline should:

- handle missing or noisy metadata
- make ranked suggestions, not just one guess
- support human review for uncertain cases
- group similar books for browsing and discovery
- keep outputs reusable across multiple catalog tasks

The upgraded pipeline is designed around those needs.

## Recommended future scope

If you want to push this further after the current upgrade, the highest-value next steps are:

1. `OCR-enhanced multimodality`

Extract text directly from the cover so subtitle typography and cover wording become predictive features.

2. `Hierarchical genre modeling`

Predict both coarse and fine labels, such as `Fiction -> Mystery` or `Nonfiction -> Politics`.

3. `Open-set rejection`

Detect books that do not belong to any trained genre and route them to review safely.

4. `Active learning`

Continuously retrain on books that the model was least certain about.

5. `Recommendation and retrieval`

Use the fused embeddings to power "related books" search, shelf organization, and duplicate detection.
