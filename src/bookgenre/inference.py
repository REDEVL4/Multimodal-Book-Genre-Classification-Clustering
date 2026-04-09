from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def predict_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    label_to_index: dict[str, int],
    device: str = "cpu",
    top_k: int = 3,
    review_threshold: float = 0.55,
) -> list[dict[str, object]]:
    inverse_labels = {index: label for label, index in label_to_index.items()}
    model.eval()
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            outputs = model(images=images, text_ids=text_ids)
            probabilities = outputs["logits"].softmax(dim=1).cpu()
            gates = outputs["gate"].detach().cpu().view(-1).tolist()
            records = batch["record"]

            for row_index in range(probabilities.size(0)):
                row = probabilities[row_index]
                confidence, predicted_index = row.max(dim=0)
                top_scores, top_indices = row.topk(min(top_k, row.size(0)))
                record = {key: values[row_index] for key, values in records.items()}
                predictions.append(
                    {
                        **record,
                        "predicted_genre": inverse_labels[int(predicted_index)],
                        "confidence": round(float(confidence), 6),
                        "needs_review": float(confidence) < review_threshold,
                        "image_weight": round(float(gates[row_index]), 6),
                        "top_k": [
                            {
                                "genre": inverse_labels[int(label_index)],
                                "score": round(float(score), 6),
                            }
                            for score, label_index in zip(top_scores.tolist(), top_indices.tolist())
                        ],
                    }
                )
    return predictions


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            outputs = model(images=images, text_ids=text_ids)
            embeddings = outputs["embeddings"].detach().cpu()
            records = batch["record"]
            for row_index in range(embeddings.size(0)):
                record = {key: values[row_index] for key, values in records.items()}
                rows.append({**record, "embedding": embeddings[row_index].tolist()})
    return rows


def write_jsonl(rows: list[dict[str, object]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
