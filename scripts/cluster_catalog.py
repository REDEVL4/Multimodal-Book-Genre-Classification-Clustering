from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bookgenre.checkpoints import load_checkpoint_bundle
from bookgenre.clustering import cluster_embeddings, find_cluster_prototypes, sample_silhouette_score
from bookgenre.data import BookMultimodalDataset, filter_records_with_images, load_book_records
from bookgenre.inference import extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster catalog embeddings from a trained checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-clusters", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--silhouette-sample-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    model, label_to_index, vocabulary, metadata = load_checkpoint_bundle(args.checkpoint_dir, device=args.device)
    records = load_book_records(args.manifest_csv)
    records, missing_records = filter_records_with_images(records, args.image_root)

    dataset = BookMultimodalDataset(
        records=records,
        image_root=args.image_root,
        label_to_index=label_to_index,
        vocabulary=vocabulary,
        max_length=int(metadata["max_length"]),
        include_author=bool(metadata["include_author"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    embedding_rows = extract_embeddings(model, loader=loader, device=args.device)
    embeddings = torch.tensor([row["embedding"] for row in embedding_rows], dtype=torch.float32)
    assignments, centroids = cluster_embeddings(embeddings, num_clusters=args.num_clusters)
    prototypes = find_cluster_prototypes(embeddings, assignments, centroids)
    silhouette = sample_silhouette_score(
        embeddings, assignments, sample_size=args.silhouette_sample_size
    )

    for row_index, row in enumerate(embedding_rows):
        row["cluster_id"] = int(assignments[row_index].item())

    output = {
        "summary": {
            "num_items": len(embedding_rows),
            "num_clusters": args.num_clusters,
            "approx_silhouette": round(float(silhouette), 6),
            "missing_images": len(missing_records),
        },
        "prototypes": {
            str(cluster_id): embedding_rows[prototype_index]
            for cluster_id, prototype_index in prototypes.items()
        },
        "items": embedding_rows,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote clustering report to {args.output_file}")


if __name__ == "__main__":
    main()
