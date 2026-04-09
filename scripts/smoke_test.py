from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bookgenre.clustering import cluster_embeddings, sample_silhouette_score
from bookgenre.model import MultimodalGenreModel


def main() -> None:
    torch.manual_seed(7)
    model = MultimodalGenreModel(
        num_classes=5,
        vocab_size=128,
        backbone_name="resnet18",
        pretrained_backbone=False,
        use_text=True,
    )
    images = torch.randn(4, 3, 224, 224)
    text_ids = torch.randint(0, 128, (4, 16))
    labels = torch.randint(0, 5, (4,))

    outputs = model(images=images, text_ids=text_ids)
    loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
    loss.backward()

    assignments, _ = cluster_embeddings(outputs["embeddings"].detach(), num_clusters=2, iterations=5)
    silhouette = sample_silhouette_score(outputs["embeddings"].detach(), assignments, sample_size=4)

    print("Smoke test passed")
    print(f"logits shape: {tuple(outputs['logits'].shape)}")
    print(f"embedding shape: {tuple(outputs['embeddings'].shape)}")
    print(f"silhouette sample: {silhouette:.4f}")


if __name__ == "__main__":
    main()
