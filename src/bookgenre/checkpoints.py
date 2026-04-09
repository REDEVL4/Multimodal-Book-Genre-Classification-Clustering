from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import TextVocabulary
from .model import MultimodalGenreModel


def save_checkpoint_bundle(
    output_dir: str | Path,
    model: MultimodalGenreModel,
    label_to_index: dict[str, int],
    vocabulary: TextVocabulary,
    metadata: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    vocabulary.save(output_dir / "vocab.json")
    (output_dir / "labels.json").write_text(json.dumps(label_to_index, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint_bundle(
    checkpoint_dir: str | Path, device: str = "cpu"
) -> tuple[MultimodalGenreModel, dict[str, int], TextVocabulary, dict[str, object]]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    label_to_index = {
        str(key): int(value)
        for key, value in json.loads((checkpoint_dir / "labels.json").read_text(encoding="utf-8")).items()
    }
    vocabulary = TextVocabulary.load(checkpoint_dir / "vocab.json")

    model = MultimodalGenreModel(
        num_classes=int(metadata["num_classes"]),
        vocab_size=int(metadata["vocab_size"]),
        backbone_name=str(metadata["backbone_name"]),
        pretrained_backbone=False,
        text_hidden_dim=int(metadata["text_hidden_dim"]),
        projection_dim=int(metadata["projection_dim"]),
        dropout=float(metadata["dropout"]),
        use_text=bool(metadata["use_text"]),
    )
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, label_to_index, vocabulary, metadata
