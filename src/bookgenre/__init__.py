"""Reusable components for multimodal book genre classification."""

from .checkpoints import load_checkpoint_bundle, save_checkpoint_bundle
from .clustering import cluster_embeddings, find_cluster_prototypes, sample_silhouette_score
from .data import (
    BookMultimodalDataset,
    BookRecord,
    TextVocabulary,
    build_label_mapping,
    compose_text,
    filter_records_with_images,
    load_book_records,
    split_records_by_label,
)
from .inference import extract_embeddings, predict_dataset
from .model import MultimodalGenreModel
from .training import TrainingConfig, train_model

__all__ = [
    "BookMultimodalDataset",
    "BookRecord",
    "MultimodalGenreModel",
    "TextVocabulary",
    "TrainingConfig",
    "build_label_mapping",
    "cluster_embeddings",
    "compose_text",
    "extract_embeddings",
    "filter_records_with_images",
    "find_cluster_prototypes",
    "load_book_records",
    "load_checkpoint_bundle",
    "predict_dataset",
    "sample_silhouette_score",
    "save_checkpoint_bundle",
    "split_records_by_label",
    "train_model",
]
